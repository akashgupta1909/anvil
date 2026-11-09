"""The tool contract. A tool is a small class with a typed argument model, a
description written for the model, and an async ``run`` that returns text.

Adding a tool (a new local capability, or an external service such as a search API)
means writing one subclass and registering it in the container; the agent loop never
changes. Tools signal failure by raising :class:`ToolError`, whose message is shown to
the model so it can correct itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from anvil.tools.workspace import Workspace


class ToolError(Exception):
    """An expected failure, reported to the model as the tool result."""


class Approver(Protocol):
    """Decides whether a side-effecting tool call may run."""

    async def approve(self, tool: str, arguments: dict[str, Any]) -> bool: ...


@dataclass(frozen=True)
class ToolContext:
    """Per-run state handed to every tool. The model never controls any of it."""

    workspace: Workspace
    approver: Approver
    max_output_chars: int
    run_timeout_seconds: float
    # Tools denied once in this run: withdrawn from the model and refused without asking
    # again (models tend to retry denied actions until they run out of steps).
    denied: set[str] = field(default_factory=lambda: set[str]())


class Tool[ArgsT: BaseModel](ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    side_effects: ClassVar[bool] = False  # writes files / executes code -> needs approval
    args_model: type[ArgsT]

    @abstractmethod
    async def run(self, args: ArgsT, ctx: ToolContext) -> str: ...

    def schema(self) -> dict[str, Any]:
        """OpenAI function-calling schema, generated from the argument model."""
        parameters = _drop_titles(self.args_model.model_json_schema())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... truncated: showing {limit} of {len(text)} characters]"


def _drop_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic adds a ``title`` to every node; they only cost the model tokens."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema
