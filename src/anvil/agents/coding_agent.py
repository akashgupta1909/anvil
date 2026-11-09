"""The agent loop: a bounded ReAct-style loop over one model and a tool registry.

Each step streams one model turn. If the model calls tools, they run and their output
goes back as (wrapped) tool messages; if not, its text is the final answer. The loop
always ends with exactly one ``FinalEvent``:

* ``completed``  — the model answered without calling a tool;
* ``max_steps``  — the step budget ran out (the last text the model wrote is returned);
* ``error``      — every LLM endpoint failed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence

from anvil.agents.planner import Planner
from anvil.config import AgentSettings
from anvil.llm.client import ChatModel, LLMUnavailableError
from anvil.models import (
    AgentEvent,
    AgentResult,
    ChatResult,
    FinalEvent,
    Message,
    PlanEvent,
    RunStatus,
    TextDeltaEvent,
    ToolCallEvent,
    ToolCallRecord,
    ToolResultEvent,
    Usage,
)
from anvil.prompts import Prompt, render, wrap_data
from anvil.tools.base import ToolContext
from anvil.tools.registry import ToolRegistry
from anvil.utils.tracing import current_trace

logger = logging.getLogger(__name__)

_EVENT_PREVIEW_CHARS = 2000


def mask_old_tool_outputs(messages: Sequence[Message], keep: int | None) -> list[Message]:
    """Observation masking: keep the last ``keep`` tool outputs verbatim and replace older
    ones with a one-line placeholder. Tool output dominates an agent's context and is
    re-sent on every step; the model can re-run a tool if it needs an old result.
    The full history is kept; only the copy sent to the model is masked."""
    tool_indexes = [i for i, m in enumerate(messages) if m.role == "tool"]
    if keep is None or len(tool_indexes) <= keep:
        return list(messages)
    masked = set(tool_indexes[: len(tool_indexes) - keep])
    calls = {c.id: c for m in messages for c in m.tool_calls}
    view: list[Message] = []
    for i, message in enumerate(messages):
        if i in masked:
            call = calls.get(message.tool_call_id or "")
            what = f"{call.name} {call.arguments[:120]}" if call else "an earlier tool call"
            placeholder = f"[Output of {what} omitted to save context; call it again if needed.]"
            message = message.model_copy(update={"content": placeholder})
        view.append(message)
    return view


class CodingAgent:
    def __init__(
        self,
        llm: ChatModel,
        registry: ToolRegistry,
        settings: AgentSettings,
        planner: Planner | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._settings = settings
        self._planner = planner
        self.system_prompt: Prompt = render("system", settings.prompts.system)

    async def run(
        self, task: str, ctx: ToolContext, history: Sequence[Message] = ()
    ) -> AgentResult:
        result: AgentResult | None = None
        async for event in self.run_stream(task, ctx, history):
            if isinstance(event, FinalEvent):
                result = event.result
        assert result is not None  # run_stream always ends with a FinalEvent
        return result

    async def run_stream(
        self, task: str, ctx: ToolContext, history: Sequence[Message] = ()
    ) -> AsyncIterator[AgentEvent]:
        trace = current_trace()
        start = time.perf_counter()
        usage_by_model: dict[str, Usage] = {}
        records: list[ToolCallRecord] = []
        plan: list[str] | None = None
        llm_calls = 0

        def finish(status: RunStatus, answer: str, steps: int, error: str | None = None):
            result = AgentResult(
                request_id=trace.trace_id,
                status=status,
                answer=answer,
                steps=steps,
                llm_calls=llm_calls,
                usage=sum(usage_by_model.values(), Usage()),
                usage_by_model=usage_by_model,
                tool_calls=records,
                plan=plan,
                error=error,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            trace.attributes.update(
                status=status.value, steps=steps, llm_calls=llm_calls, **result.usage.model_dump()
            )
            return FinalEvent(result=result)

        def account(model: str | None, usage: Usage) -> None:
            if model:
                usage_by_model[model] = usage_by_model.get(model, Usage()) + usage

        messages = [Message(role="system", content=self.system_prompt.text), *history]
        trace.attributes.update(system_prompt=self.system_prompt.ref)

        if self._planner:
            plan, plan_usage, plan_model = await self._planner.plan(task)
            account(plan_model, plan_usage)
            llm_calls += 1
            if plan:
                yield PlanEvent(steps=plan)
                task = (
                    f"{task}\n\nA planning step suggested this approach (adapt it if the "
                    "workspace shows otherwise):\n" + "\n".join(f"- {s}" for s in plan)
                )
        messages.append(Message(role="user", content=task))

        last_text = ""
        for step in range(1, self._settings.max_steps + 1):
            reply: ChatResult | None = None
            tools = self._registry.schemas(exclude=ctx.denied)
            try:
                view = mask_old_tool_outputs(messages, self._settings.keep_tool_outputs)
                async for chunk in self._llm.stream(view, tools=tools):
                    if isinstance(chunk, str):
                        yield TextDeltaEvent(step=step, text=chunk)
                    else:
                        reply = chunk
            except LLMUnavailableError as exc:
                logger.error("llm unavailable", extra={"err": str(exc)})
                yield finish(
                    RunStatus.ERROR,
                    "The language model is unavailable right now; please retry later.",
                    step - 1,
                    error=str(exc),
                )
                return
            assert reply is not None  # stream always ends with a ChatResult
            llm_calls += 1
            account(reply.model, reply.usage)
            messages.append(reply.message)
            last_text = reply.message.content or last_text

            if not reply.message.tool_calls:
                yield finish(RunStatus.COMPLETED, reply.message.content or "", step)
                return

            for call in reply.message.tool_calls:
                yield ToolCallEvent(step=step, name=call.name, arguments=call.arguments)
                outcome = await self._registry.execute(call, ctx)
                records.append(outcome.record)
                yield ToolResultEvent(
                    step=step,
                    name=call.name,
                    ok=outcome.record.ok,
                    output=outcome.content[:_EVENT_PREVIEW_CHARS],
                    duration_ms=outcome.record.duration_ms,
                )
                content = (
                    wrap_data("tool_output", outcome.content, tool=call.name)
                    if self._settings.wrap_tool_output
                    else outcome.content
                )
                messages.append(Message(role="tool", content=content, tool_call_id=call.id))

        steps = self._settings.max_steps
        answer = last_text or f"Stopped after {steps} steps without reaching a final answer."
        yield finish(RunStatus.MAX_STEPS, answer, steps)
