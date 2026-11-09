import argparse
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from call_function import available_functions, call_function
from prompts import system_prompt

MAX_ITERATIONS = 20


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str


GROQ = Provider(
    name="Groq",
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
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


def ask(provider: Provider, messages: list):
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


def ask_with_fallback(providers: list[Provider], messages: list):
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


def build_messages(user_prompt: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def assistant_message(message: ChatCompletionMessage) -> dict:
    """Keep only the standard fields, since providers add extras (e.g. reasoning_details)
    that other providers reject when the history is re-sent."""
    return message.model_dump(
        include={"role", "content", "tool_calls"}, exclude_none=True
    )


def print_usage(response: ChatCompletion) -> None:
    print(f"Prompt Tokens: {response.usage.prompt_tokens}")
    print(f"Output Tokens: {response.usage.completion_tokens}")


def run_agent(providers: list[Provider], user_prompt: str, verbose: bool) -> None:
    messages = build_messages(user_prompt)
    if verbose:
        print(f"User Prompt: {user_prompt}")

    for _ in range(MAX_ITERATIONS):
        response = ask_with_fallback(providers, messages)
        if verbose:
            print_usage(response)

        message = response.choices[0].message
        messages.append(assistant_message(message))

        if not message.tool_calls:
            print(message.content)
            return

        for tool_call in message.tool_calls:
            result_message = call_function(tool_call, verbose)
            if not result_message["content"]:
                raise RuntimeError(
                    f"Function {tool_call.function.name} returned no content"
                )
            if verbose:
                print(f"-> {result_message['content']}")
            messages.append(result_message)

    print(
        f"Error: agent did not produce a final response after {MAX_ITERATIONS} iterations",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    load_dotenv()
    args = parse_args()
    providers = [OPENROUTER] if args.open_router else [GROQ, OPENROUTER]
    run_agent(providers, args.user_prompt, args.verbose)


if __name__ == "__main__":
    main()
