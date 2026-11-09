"""Shared data types used across layers (LLM, tools, agent, API, evaluation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

# --- Conversation ---------------------------------------------------------------------


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # raw JSON text as produced by the model; validated by the tool registry


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        """Serialise with standard fields only. Providers add extras (e.g. reasoning
        details) that other providers reject when the history is re-sent on fallback."""
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ChatResult(BaseModel):
    message: Message
    usage: Usage
    provider: str
    model: str


# --- Agent run ------------------------------------------------------------------------


class RunStatus(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    ERROR = "error"


ToolErrorType = Literal["unknown_tool", "bad_arguments", "denied", "tool_error"]


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] | None  # None when the model sent unparsable JSON
    ok: bool
    error_type: ToolErrorType | None = None  # unknown_tool / bad_arguments = model mistakes
    error: str | None = None
    duration_ms: float


class AgentResult(BaseModel):
    request_id: str
    status: RunStatus
    answer: str
    steps: int
    llm_calls: int
    usage: Usage
    usage_by_model: dict[str, Usage]
    tool_calls: list[ToolCallRecord]
    changed_files: list[str] = []
    plan: list[str] | None = None
    error: str | None = None
    duration_ms: float


# --- Streamed events (one SSE event each) ---------------------------------------------


class PlanEvent(BaseModel):
    type: Literal["plan"] = "plan"
    steps: list[str]


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    step: int
    text: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    step: int
    name: str
    arguments: str


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    step: int
    name: str
    ok: bool
    output: str
    duration_ms: float


class FinalEvent(BaseModel):
    type: Literal["final"] = "final"
    result: AgentResult


type AgentEvent = PlanEvent | TextDeltaEvent | ToolCallEvent | ToolResultEvent | FinalEvent
