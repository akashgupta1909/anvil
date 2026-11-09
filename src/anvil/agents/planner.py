"""Optional planning step on the cheap ``fast`` model. Fails open to "no plan"."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from anvil.llm.client import (
    ChatModel,
    LLMUnavailableError,
    StructuredOutputError,
    complete_structured,
)
from anvil.models import Message, Usage
from anvil.prompts import render, wrap_data
from anvil.tools.registry import ToolRegistry
from anvil.utils.tracing import current_trace

logger = logging.getLogger(__name__)


class Plan(BaseModel):
    steps: list[str] = Field(description="1 to 5 short, concrete steps.")


class Planner:
    def __init__(self, llm: ChatModel, registry: ToolRegistry, prompt_version: str) -> None:
        self._llm = llm
        tools = "\n".join(f"- {t.name}: {t.description}" for t in registry.tools)
        self._prompt = render("planner", prompt_version, tools=tools)

    async def plan(self, task: str) -> tuple[list[str] | None, Usage, str | None]:
        """Returns (steps or None, usage, model used)."""
        messages = [
            Message(role="system", content=self._prompt.text),
            Message(role="user", content=wrap_data("task", task)),
        ]
        with current_trace().span("plan", prompt=self._prompt.ref) as span:
            try:
                plan, result = await complete_structured(self._llm, messages, Plan)
            except (LLMUnavailableError, StructuredOutputError) as exc:
                span.status, span.error = "error", str(exc)[:300]
                logger.warning("planner failed; continuing without a plan", extra={"err": str(exc)})
                return None, Usage(), None
            steps = [s.strip() for s in plan.steps if s.strip()][:5]
            span.set(steps=len(steps))
            return steps or None, result.usage, result.model
