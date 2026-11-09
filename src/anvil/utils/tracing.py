"""Per-request tracing.

A ``Trace`` holds a flat list of timed spans (agent run, LLM calls, tool calls). The
active trace lives in a context variable, so the LLM client and tool registry record
spans without the trace being threaded through every signature. Span and attribute
names follow the OpenTelemetry GenAI conventions (``chat``, ``execute_tool``,
``gen_ai.*``) so an OTel exporter can be added later without renaming anything.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    start_offset_ms: float
    attributes: dict[str, Any] = field(default_factory=lambda: {})
    duration_ms: float = 0.0
    status: str = "ok"
    error: str | None = None

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    attributes: dict[str, Any] = field(default_factory=lambda: {})
    spans: list[Span] = field(default_factory=lambda: [])
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Span]:
        start = time.perf_counter()
        span = Span(name, round((start - self._t0) * 1000, 1), dict(attributes))
        self.spans.append(span)
        try:
            yield span
        except BaseException as exc:
            span.status, span.error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_t0")
        return data


_current: ContextVar[Trace | None] = ContextVar("trace", default=None)


def current_trace() -> Trace:
    """The active trace, or a throwaway one so callers never need a None check."""
    return _current.get() or Trace()


@contextmanager
def activate(trace: Trace) -> Generator[Trace]:
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


class TraceSink:
    """Persists finished traces as one JSON file per request."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def write(self, trace: Trace) -> Path | None:
        path = self._dir / f"{trace.trace_id}.json"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
        except OSError:
            # Tracing must never fail a request.
            logger.warning("could not write trace", exc_info=True, extra={"path": str(path)})
            return None
        return path
