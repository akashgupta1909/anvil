"""HTTP request/response bodies. Agent results reuse the shared domain models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=8000, description="What the agent should do.")
    session_id: str | None = Field(
        None, description="Continue a session (its workspace and memory). Omit for one-shot."
    )
    allow_writes: bool = Field(
        False, description="Allow side-effecting tools (write_file, run_python_file)."
    )


class SessionResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_endpoints: dict[str, list[str]]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None


class ErrorResponse(BaseModel):
    error: ErrorBody
