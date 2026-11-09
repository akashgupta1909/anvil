"""LLM-as-judge for final answers, run on a different model family than the agent."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from anvil.llm.client import (
    ChatModel,
    LLMUnavailableError,
    StructuredOutputError,
    complete_structured,
)
from anvil.models import AgentResult, Message
from anvil.prompts import render, wrap_data

logger = logging.getLogger(__name__)


class Verdict(BaseModel):
    passed: bool
    reason: str


class Judge:
    def __init__(self, llm: ChatModel, prompt_version: str) -> None:
        self._llm = llm
        self._prompt = render("judge", prompt_version)

    async def grade(self, task: str, rubric: str, result: AgentResult) -> Verdict | None:
        """Returns None if the judge could not produce a verdict (fails open to 'ungraded',
        never to 'passed')."""
        calls = [
            {"tool": c.name, "arguments": c.arguments, "ok": c.ok, "error": c.error}
            for c in result.tool_calls
        ]
        user = "\n\n".join(
            [
                wrap_data("task", task),
                f"Rubric: {rubric}",
                wrap_data("tool_calls", json.dumps(calls)[:4000]),
                wrap_data("answer", result.answer[:6000]),
            ]
        )
        messages = [
            Message(role="system", content=self._prompt.text),
            Message(role="user", content=user),
        ]
        try:
            verdict, _ = await complete_structured(self._llm, messages, Verdict)
        except (LLMUnavailableError, StructuredOutputError) as exc:
            logger.warning("judge failed", extra={"err": str(exc)[:300]})
            return None
        return verdict
