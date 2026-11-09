import argparse
import json
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

from call_function import available_functions
from prompts import system_prompt


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str


GROQ = Provider(
    name="Groq",
    base_url="https://api.groq.com/openai/v1",
    model="qwen/qwen3.8-27b",
    api_key_env="GROQ_API_KEY",
)
OPENROUTER = Provider(
    name="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    api_key_env="OPENROUTER_API_KEY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument(
        "--open_router",
        action="store_true",
        help="Use OpenRouter directly instead of Groq (with OpenRouter as fallback)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    if not args.user_prompt.strip():
        parser.error("User prompt should not be empty")
    return args


def create_client(provider: Provider) -> OpenAI:
    api_key = os.environ.get(provider.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{provider.name} API key is missing ({provider.api_key_env})"
        )
    return OpenAI(base_url=provider.base_url, api_key=api_key)


def ask(provider: Provider, messages: list[dict[str, str]]):
    client = create_client(provider)
    response = client.chat.completions.create(
        model=provider.model,
        messages=messages,
        tools=available_functions,
        temperature=0,
    )
    if response.usage is None:
        raise RuntimeError(
            f"{provider.name} response has no usage data, indicating the request failed"
        )
    return response


def ask_with_fallback(providers: list[Provider], messages: list[dict[str, str]]):
    for i, provider in enumerate(providers):
        try:
            return ask(provider, messages)
        except Exception as e:
            if i == len(providers) - 1:
                raise
            print(
                f"{provider.name} failed ({e}); falling back to {providers[i + 1].name}",
                file=sys.stderr,
            )


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def print_output(response: ChatCompletion, user_prompt: str, verbose: bool = False):
    if verbose:
        print(f"User Prompt: {user_prompt}")
        print(f"Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"Output Tokens: {response.usage.completion_tokens}")

    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_args = json.loads(tool_call.function.arguments or "{}")
            print(f"Calling function: {tool_call.function.name}({function_args})")
    else:
        print(message.content)


def main() -> None:
    load_dotenv()
    args = parse_args()
    providers = [OPENROUTER] if args.open_router else [GROQ, OPENROUTER]

    response = ask_with_fallback(providers, build_messages(args.user_prompt))
    print_output(response, args.user_prompt, args.verbose)


if __name__ == "__main__":
    main()
