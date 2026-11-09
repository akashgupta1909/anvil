"""Tool registry: schemas for the model, and safe execution of the model's calls.

``execute`` never raises for anything the model got wrong (unknown tool, malformed
JSON, invalid arguments, a denied call, a tool error): each becomes a tool result the
model can read and recover from. Only genuine bugs propagate.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from anvil.models import ToolCall, ToolCallRecord, ToolErrorType
from anvil.tools.base import Tool, ToolContext, ToolError
from anvil.utils.tracing import current_trace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolOutcome:
    record: ToolCallRecord
    content: str  # what the model sees


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]]) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    @property
    def tools(self) -> list[Tool[Any]]:
        return list(self._tools.values())

    def schemas(self, exclude: set[str] | frozenset[str] = frozenset()) -> list[dict[str, Any]]:
        return [tool.schema() for name, tool in self._tools.items() if name not in exclude]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        start = time.perf_counter()
        with current_trace().span(
            "execute_tool", **{"gen_ai.tool.name": call.name, "gen_ai.tool.call.id": call.id}
        ) as span:
            arguments, content, failure = await self._execute(call, ctx)
            if failure:
                span.status, span.error = "error", f"{failure[0]}: {failure[1]}"
            span.set(output_chars=len(content))
        record = ToolCallRecord(
            name=call.name,
            arguments=arguments,
            ok=failure is None,
            error_type=failure[0] if failure else None,
            error=failure[1] if failure else None,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return ToolOutcome(record, content)

    async def _execute(
        self, call: ToolCall, ctx: ToolContext
    ) -> tuple[dict[str, Any] | None, str, tuple[ToolErrorType, str] | None]:
        """Returns (parsed arguments, content for the model, (error type, message) or None)."""
        tool = self._tools.get(call.name)
        if tool is None:
            error = f"unknown tool {call.name!r}; available: {', '.join(self._tools)}"
            return None, f"Error: {error}", ("unknown_tool", error)
        try:
            raw = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            error = f"arguments are not valid JSON ({exc.msg})"
            return None, f"Error: {error}", ("bad_arguments", error)
        arguments = cast(dict[str, Any], raw) if isinstance(raw, dict) else None
        try:
            args = tool.args_model.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()
            )
            error = f"invalid arguments: {problems}"
            return arguments, f"Error: {error}", ("bad_arguments", error)
        if tool.side_effects and (
            tool.name in ctx.denied or not await ctx.approver.approve(tool.name, arguments or {})
        ):
            ctx.denied.add(tool.name)
            error = f"{tool.name} was not approved and is disabled for the rest of this task"
            content = (
                f"Error: {error}. Do not retry. In your answer, say plainly that the action "
                "was not permitted, so it was not done, then describe what you would have done."
            )
            return arguments, content, ("denied", error)
        try:
            return arguments, await tool.run(args, ctx), None
        except ToolError as exc:
            return arguments, f"Error: {exc}", ("tool_error", str(exc))
        except OSError as exc:
            logger.warning("tool failed", extra={"tool": tool.name}, exc_info=True)
            return arguments, f"Error: {exc.strerror or exc}", ("tool_error", str(exc))
