"""OpenAI-compatible chat client with ordered provider fallback.

Retry policy (the SDK's own retries are disabled so this is the only one): a failing
endpoint immediately hands over to the next one in the chain, because waiting out a
rate limit (``Retry-After`` is often 20s+ on free tiers) is slower than another
provider or model. Only when every endpoint has failed with a retryable error (429, 5xx,
timeout, connection) does the client wait (the larger of ``Retry-After`` and exponential
back-off, capped) and walk the chain again. Each attempt is a ``chat`` span on the trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import openai
from openai import AsyncOpenAI, Omit
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel, ValidationError

from anvil.llm.reasoning import ReasoningFilter, strip_reasoning
from anvil.models import ChatResult, Message, ToolCall, Usage
from anvil.utils.tracing import current_trace

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
_MAX_COOLDOWN_SECONDS = 600.0


class LLMUnavailableError(RuntimeError):
    """Every endpoint in the fallback chain failed."""


class StructuredOutputError(RuntimeError):
    """The model's reply did not validate against the requested schema."""


class _EmptyResponseError(RuntimeError):
    """A 200 response without choices (some routers report errors this way)."""


@dataclass(frozen=True)
class Endpoint:
    provider: str
    model: str
    client: AsyncOpenAI


class ChatModel(Protocol):
    """What the agent, planner and judge need from an LLM. Tests provide fakes."""

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...

    def stream(
        self, messages: Sequence[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str | ChatResult]: ...


_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,  # includes timeouts
    openai.InternalServerError,
    _EmptyResponseError,
)


class FallbackLLM:
    """A model *role* (main, fast, judge) backed by an ordered list of endpoints."""

    def __init__(
        self,
        role: str,
        endpoints: Sequence[Endpoint],
        temperature: float,
        *,
        retry_rounds: int = 0,
        max_retry_wait: float = 30.0,
    ) -> None:
        self.role = role
        self.endpoints = list(endpoints)
        self._temperature = temperature
        self._retry_rounds = retry_rounds
        self._max_retry_wait = max_retry_wait
        # Circuit breaker: endpoint -> monotonic time until which it is skipped, set from
        # a 429's Retry-After so an exhausted quota is not retried first on every call.
        self._cooldown: dict[Endpoint, float] = {}

    def _attempts(self) -> _Attempts:
        now = time.monotonic()
        ready = [e for e in self.endpoints if self._cooldown.get(e, 0) <= now]
        # If everything is cooling down, try them all anyway rather than fail outright.
        return _Attempts(
            self.role,
            ready or self.endpoints,
            self._retry_rounds + 1,
            self._max_retry_wait,
            self._cooldown,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        attempts = self._attempts()
        async for endpoint in attempts:
            with current_trace().span("chat", **self._span_attrs(endpoint)) as span:
                try:
                    response = await endpoint.client.chat.completions.create(
                        model=endpoint.model,
                        messages=_messages(messages),
                        tools=_tools(tools),
                        response_format=cast(Any, response_format) or openai.omit,
                        temperature=self._temperature,
                    )
                    if not response.choices:
                        raise _EmptyResponseError("response has no choices")
                except (openai.APIError, _EmptyResponseError) as exc:
                    span.status, span.error = "error", attempts.failed(endpoint, exc)
                    continue
                choice = response.choices[0].message
                usage = Usage(
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0,
                )
                message = Message(
                    role="assistant",
                    content=strip_reasoning(choice.content or "") or None,
                    tool_calls=[
                        ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
                        for c in choice.tool_calls or []
                        if c.type == "function"
                    ],
                )
                span.set(**_usage_attrs(usage), **{"gen_ai.response.model": response.model})
                return ChatResult(
                    message=message, usage=usage, provider=endpoint.provider, model=endpoint.model
                )
        raise LLMUnavailableError(attempts.summary())

    async def stream(
        self, messages: Sequence[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str | ChatResult]:
        """Yield text deltas, then one final :class:`ChatResult`.

        Falls back to the next endpoint only if the failure happens before any text
        was yielded; after that a partial answer cannot be retracted, so it raises.
        """
        attempts = self._attempts()
        async for endpoint in attempts:
            emitted = False
            with current_trace().span("chat", **self._span_attrs(endpoint), stream=True) as span:
                try:
                    start = time.perf_counter()
                    text_filter = ReasoningFilter()
                    acc = _StreamAccumulator()
                    chunks = await endpoint.client.chat.completions.create(
                        model=endpoint.model,
                        messages=_messages(messages),
                        tools=_tools(tools),
                        temperature=self._temperature,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    async for chunk in chunks:
                        acc.add(chunk)
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta and (visible := text_filter.feed(delta)):
                            if not emitted:
                                span.set(time_to_first_token_ms=_ms_since(start))
                            emitted = True
                            yield visible
                    if tail := text_filter.flush():
                        emitted = True
                        yield tail
                    if not acc.received_choices:
                        raise _EmptyResponseError("stream has no choices")
                except (openai.APIError, _EmptyResponseError) as exc:
                    span.status, span.error = "error", attempts.failed(endpoint, exc)
                    if emitted:
                        raise LLMUnavailableError(
                            f"{endpoint.provider} failed mid-stream: {span.error}"
                        ) from exc
                    continue
                result = acc.result(endpoint)
                span.set(**_usage_attrs(result.usage))
                yield result
                return
        raise LLMUnavailableError(attempts.summary())

    def _span_attrs(self, endpoint: Endpoint) -> dict[str, Any]:
        return {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": endpoint.provider,
            "gen_ai.request.model": endpoint.model,
            "role": self.role,
        }


class _Attempts:
    """Yields endpoints in chain order, for up to ``rounds`` passes. Between passes it
    sleeps, but only if some failure in the last pass was retryable."""

    def __init__(
        self,
        role: str,
        endpoints: list[Endpoint],
        rounds: int,
        max_wait: float,
        cooldown: dict[Endpoint, float],
    ) -> None:
        self._role = role
        self._endpoints = endpoints
        self._rounds = rounds
        self._max_wait = max_wait
        self._cooldown = cooldown
        self._errors: list[str] = []
        self._retryable = False
        self._retry_after = 0.0

    async def __aiter__(self) -> AsyncIterator[Endpoint]:
        for round_ in range(self._rounds):
            if round_:
                if not self._retryable:
                    return
                delay = min(self._max_wait, max(self._retry_after, 2.0 ** (round_ - 1)))
                logger.warning(
                    "all llm endpoints failed; retrying chain",
                    extra={"role": self._role, "round": round_, "delay_s": delay},
                )
                await asyncio.sleep(delay)
            self._retryable, self._retry_after = False, 0.0
            for endpoint in self._endpoints:
                yield endpoint

    def failed(self, endpoint: Endpoint, exc: Exception) -> str:
        description = _describe(exc)
        self._errors.append(f"{endpoint.provider}/{endpoint.model}: {description}")
        if isinstance(exc, _RETRYABLE) or _is_failed_generation(exc):
            self._retryable = True
        if isinstance(exc, openai.RateLimitError):
            wait = _retry_after_seconds(exc)
            self._retry_after = max(self._retry_after, wait)
            if wait > 0:
                self._cooldown[endpoint] = time.monotonic() + min(wait, _MAX_COOLDOWN_SECONDS)
        logger.warning(
            "llm endpoint failed, falling back",
            extra={
                "role": self._role,
                "endpoint": f"{endpoint.provider}/{endpoint.model}",
                "err": description,
            },
        )
        return description

    def summary(self) -> str:
        if not self._endpoints:
            return f"no LLM endpoint configured for role {self._role!r} (missing API keys?)"
        return f"all endpoints failed for role {self._role!r}: " + "; ".join(self._errors)


async def complete_structured[T: BaseModel](
    llm: ChatModel, messages: Sequence[Message], schema: type[T]
) -> tuple[T, ChatResult]:
    """Ask for JSON matching ``schema`` (strict ``json_schema`` response format) and
    validate it. Providers that ignore the format are still caught by validation."""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": _strict_schema(schema),
        },
    }
    result = await llm.complete(messages, response_format=response_format)
    raw = _FENCE_RE.sub("", result.message.content or "").strip()
    try:
        return schema.model_validate_json(raw), result
    except ValidationError as exc:
        raise StructuredOutputError(
            f"{schema.__name__} validation failed: {exc.error_count()} error(s); raw={raw[:200]!r}"
        ) from exc


def _strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Strict mode requires every property to be required and no extra properties."""
    json_schema = schema.model_json_schema()
    json_schema["additionalProperties"] = False
    json_schema["required"] = list(json_schema.get("properties", {}))
    return json_schema


class _StreamAccumulator:
    """Reassembles a streamed completion: text, tool calls split across chunks, usage."""

    def __init__(self) -> None:
        self.received_choices = False
        self._text: list[str] = []
        self._calls: dict[int, dict[str, str]] = {}
        self._usage = Usage()

    def add(self, chunk: ChatCompletionChunk) -> None:
        if chunk.usage:
            self._usage = Usage(
                input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens
            )
        if not chunk.choices:
            return
        self.received_choices = True
        delta = chunk.choices[0].delta
        if delta.content:
            self._text.append(delta.content)
        for call in delta.tool_calls or []:
            slot = self._calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
            if call.id:
                slot["id"] = call.id
            if call.function and call.function.name:
                slot["name"] += call.function.name
            if call.function and call.function.arguments:
                slot["arguments"] += call.function.arguments

    def result(self, endpoint: Endpoint) -> ChatResult:
        calls = [
            ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"])
            for i, c in sorted(self._calls.items())
        ]
        message = Message(
            role="assistant",
            content=strip_reasoning("".join(self._text)) or None,
            tool_calls=calls,
        )
        return ChatResult(
            message=message, usage=self._usage, provider=endpoint.provider, model=endpoint.model
        )


def _messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    return [cast(ChatCompletionMessageParam, m.to_openai()) for m in messages]


def _tools(tools: list[dict[str, Any]] | None) -> list[ChatCompletionToolUnionParam] | Omit:
    return [cast(ChatCompletionToolUnionParam, t) for t in tools] if tools else openai.omit


def _usage_attrs(usage: Usage) -> dict[str, int]:
    return {
        "gen_ai.usage.input_tokens": usage.input_tokens,
        "gen_ai.usage.output_tokens": usage.output_tokens,
    }


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _is_failed_generation(exc: Exception) -> bool:
    """Groq rejects (HTTP 400) a sampled reply that fails strict JSON-schema validation.
    The request itself is fine, so a resample usually succeeds: treat it as retryable."""
    return isinstance(exc, openai.BadRequestError) and exc.code == "json_validate_failed"


def _retry_after_seconds(exc: openai.APIStatusError) -> float:
    try:
        return float(exc.response.headers.get("retry-after", 0))
    except ValueError:
        return 0.0


def _describe(exc: Exception) -> str:
    if isinstance(exc, openai.APIStatusError):
        body = cast(Any, exc.body)
        detail = json.dumps(body)[:200] if body else exc.message[:200]
        return f"HTTP {exc.status_code}: {detail}"
    return f"{type(exc).__name__}: {exc}"[:200]
