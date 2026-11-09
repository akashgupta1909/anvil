"""The real OpenAI SDK against an in-process HTTP transport (no network).

openai>=3 is built on ``httpx2`` (not ``httpx``), so the mock transport comes from there.
"""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from anvil.llm.client import (
    Endpoint,
    FallbackLLM,
    LLMUnavailableError,
    StructuredOutputError,
    complete_structured,
)
from anvil.llm.reasoning import ReasoningFilter, strip_reasoning
from anvil.models import ChatResult, Message
from anvil.utils.tracing import Trace, activate

Handler = Callable[[httpx2.Request], httpx2.Response]
USER = [Message(role="user", content="hi")]


def endpoint(name: str, handler: Handler) -> Endpoint:
    client = AsyncOpenAI(
        base_url=f"http://{name}.test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return Endpoint(name, f"{name}-model", client)


def completion(
    content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
) -> httpx2.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx2.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "served-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    )


def sse(*chunks: dict[str, Any]) -> httpx2.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


def chunk(
    delta: dict[str, Any] | None = None, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
        if delta is not None
        else [],
        "usage": usage,
    }


def fail(status: int = 500) -> Handler:
    return lambda _: httpx2.Response(status, json={"error": {"message": "boom"}})


async def test_falls_back_to_next_endpoint() -> None:
    llm = FallbackLLM("main", [endpoint("a", fail()), endpoint("b", lambda _: completion("ok"))], 0)
    with activate(Trace()) as trace:
        result = await llm.complete(USER)
    assert (result.provider, result.message.content) == ("b", "ok")
    assert result.usage.input_tokens == 7
    assert [s.status for s in trace.spans] == ["error", "ok"]
    assert "HTTP 500" in (trace.spans[0].error or "")


async def test_all_endpoints_failing_raises() -> None:
    llm = FallbackLLM("main", [endpoint("a", fail(429)), endpoint("b", fail(503))], 0)
    with pytest.raises(LLMUnavailableError, match=r"a/a-model: HTTP 429.*b/b-model: HTTP 503"):
        await llm.complete(USER)


async def test_retries_the_chain_after_rate_limits() -> None:
    calls: list[str] = []

    def flaky(request: httpx2.Request) -> httpx2.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx2.Response(429, headers={"retry-after": "0.01"}, json={"error": {}})
        return completion("ok")

    llm = FallbackLLM("main", [endpoint("a", flaky)], 0, retry_rounds=1, max_retry_wait=0.05)
    assert (await llm.complete(USER)).message.content == "ok"
    assert len(calls) == 2


async def test_failed_json_generation_is_retried() -> None:
    calls: list[int] = []

    def flaky(_: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            body = {"error": {"message": "Failed to generate JSON", "code": "json_validate_failed"}}
            return httpx2.Response(400, json=body)
        return completion('{"value": 1}')

    llm = FallbackLLM("judge", [endpoint("a", flaky)], 0, retry_rounds=1, max_retry_wait=0)
    answer, _ = await complete_structured(llm, USER, Answer)
    assert answer.value == 1


async def test_non_retryable_errors_do_not_start_another_round() -> None:
    calls: list[int] = []

    def unauthorized(_: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        return httpx2.Response(401, json={"error": {"message": "bad key"}})

    llm = FallbackLLM("main", [endpoint("a", unauthorized)], 0, retry_rounds=3, max_retry_wait=0)
    with pytest.raises(LLMUnavailableError, match="HTTP 401"):
        await llm.complete(USER)
    assert len(calls) == 1


async def test_no_endpoints_raises_helpful_error() -> None:
    with pytest.raises(LLMUnavailableError, match="missing API keys"):
        await FallbackLLM("judge", [], 0).complete(USER)


async def test_strips_reasoning_and_parses_tool_calls() -> None:
    calls = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
    llm = FallbackLLM(
        "main", [endpoint("a", lambda _: completion("<think>hmm</think>Answer", calls))], 0
    )
    result = await llm.complete(USER)
    assert result.message.content == "Answer"
    assert result.message.tool_calls[0].name == "read_file"


async def test_sends_only_standard_message_fields() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return completion("ok")

    llm = FallbackLLM("main", [endpoint("a", handler)], 0.5)
    await llm.complete([Message(role="tool", content="x", tool_call_id="c1")])
    assert seen[0]["messages"] == [{"role": "tool", "content": "x", "tool_call_id": "c1"}]
    assert seen[0]["temperature"] == 0.5
    assert "tools" not in seen[0]


async def test_stream_assembles_text_tool_calls_and_usage() -> None:
    response = sse(
        chunk({"role": "assistant", "content": "<thi"}),
        chunk({"content": "nk>secret</think>Hel"}),
        chunk({"content": "lo"}),
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_", "arguments": '{"pa'},
                    }
                ]
            }
        ),
        chunk(
            {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": 'th": "a"}'}}]}
        ),
        chunk(usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}),
    )
    llm = FallbackLLM("main", [endpoint("a", lambda _: response)], 0)
    out = [c async for c in llm.stream(USER)]
    deltas = [c for c in out if isinstance(c, str)]
    final = out[-1]
    assert "".join(deltas) == "Hello"
    assert isinstance(final, ChatResult)
    assert final.message.content == "Hello"
    assert final.message.tool_calls[0].name == "read_file"
    assert json.loads(final.message.tool_calls[0].arguments) == {"path": "a"}
    assert final.usage.input_tokens == 11


async def test_stream_falls_back_before_first_token() -> None:
    ok = sse(chunk({"content": "fine"}))
    llm = FallbackLLM("main", [endpoint("a", fail()), endpoint("b", lambda _: ok)], 0)
    out = [c async for c in llm.stream(USER)]
    final = out[-1]
    assert isinstance(final, ChatResult)
    assert final.provider == "b"


async def test_empty_choices_trigger_fallback() -> None:
    empty = httpx2.Response(
        200,
        json={"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []},
    )
    llm = FallbackLLM(
        "main", [endpoint("a", lambda _: empty), endpoint("b", lambda _: completion("ok"))], 0
    )
    assert (await llm.complete(USER)).provider == "b"


class Answer(BaseModel):
    value: int


async def test_structured_output_validates() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return completion('```json\n{"value": 42}\n```')

    llm = FallbackLLM("fast", [endpoint("a", handler)], 0)
    answer, _ = await complete_structured(llm, USER, Answer)
    assert answer.value == 42
    response_format = seen[0]["response_format"]
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


async def test_structured_output_rejects_invalid_json() -> None:
    llm = FallbackLLM("fast", [endpoint("a", lambda _: completion("not json"))], 0)
    with pytest.raises(StructuredOutputError):
        await complete_structured(llm, USER, Answer)


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["<think>a</think>b"], "b"),
        (["x<th", "ink>y</th", "ink>z"], "xz"),
        (["a < b"], "a < b"),
        (["<think>never closed"], ""),
        (["tail <"], "tail <"),
    ],
)
def test_reasoning_filter_handles_split_tags(chunks: list[str], expected: str) -> None:
    f = ReasoningFilter()
    assert "".join(f.feed(c) for c in chunks) + f.flush() == expected


def test_strip_reasoning() -> None:
    assert strip_reasoning("<think>\nplan\n</think>\n\nDone") == "Done"
