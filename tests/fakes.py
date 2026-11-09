"""Test doubles and helpers. Nothing here touches the network: LLMs are scripted fakes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from anvil.llm.client import LLMUnavailableError
from anvil.models import ChatResult, Message, ToolCall, Usage
from anvil.tools.approval import AllowAll
from anvil.tools.base import Approver, ToolContext
from anvil.tools.workspace import Workspace

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "calculator"


def reply(content: str | None = None, *calls: tuple[str, str]) -> Message:
    """An assistant message; ``calls`` are (tool name, JSON arguments) pairs."""
    return Message(
        role="assistant",
        content=content,
        tool_calls=[
            ToolCall(id=f"call_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)
        ],
    )


class FakeLLM:
    """Replays scripted replies; an Exception in the script is raised instead."""

    def __init__(self, script: Sequence[Message | Exception], model: str = "fake-model") -> None:
        self.script = list(script)
        self.model = model
        self.requests: list[list[Message]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    def _next(self, messages: Sequence[Message], tools: list[dict[str, Any]] | None) -> ChatResult:
        self.requests.append(list(messages))
        self.tools_seen.append(tools)
        if not self.script:
            raise LLMUnavailableError("fake script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(
            message=item,
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="fake",
            model=self.model,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        return self._next(messages, tools)

    async def stream(
        self, messages: Sequence[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str | ChatResult]:
        result = self._next(messages, tools)
        text = result.message.content or ""
        if text:
            yield text[: len(text) // 2]
            yield text[len(text) // 2 :]
        yield result


def make_ctx(
    workspace: Workspace, approver: Approver | None = None, **overrides: Any
) -> ToolContext:
    params: dict[str, Any] = {"max_output_chars": 10_000, "run_timeout_seconds": 10} | overrides
    return ToolContext(workspace=workspace, approver=approver or AllowAll(), **params)
