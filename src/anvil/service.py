"""Application service: runs one agent request end to end.

Shared by the CLI and the API. It owns everything around the agent loop: the
per-request trace (created, activated, persisted), which workspace the tools see
(a session's, a caller-supplied directory, or a disposable copy), conversation
memory, and per-session locking.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import nullcontext

from anvil.agents.coding_agent import CodingAgent
from anvil.config import Settings
from anvil.memory.sessions import Session, SessionStore, Turn
from anvil.models import AgentEvent, AgentResult, FinalEvent, RunStatus
from anvil.tools.base import Approver, ToolContext
from anvil.tools.changes import changed_files
from anvil.tools.workspace import Workspace, WorkspaceManager
from anvil.utils.tracing import Trace, TraceSink, activate

logger = logging.getLogger(__name__)


class SessionBusyError(RuntimeError):
    """A request is already running in this session (its workspace is in use)."""


class AgentService:
    def __init__(
        self,
        agent: CodingAgent,
        sessions: SessionStore,
        workspaces: WorkspaceManager,
        trace_sink: TraceSink,
        settings: Settings,
    ) -> None:
        self.agent = agent
        self.sessions = sessions
        self._workspaces = workspaces
        self._trace_sink = trace_sink
        self._settings = settings

    async def run(
        self,
        task: str,
        approver: Approver,
        *,
        session_id: str | None = None,
        workspace: Workspace | None = None,
        request_id: str | None = None,
    ) -> AgentResult:
        result: AgentResult | None = None
        async for event in self.run_stream(
            task, approver, session_id=session_id, workspace=workspace, request_id=request_id
        ):
            if isinstance(event, FinalEvent):
                result = event.result
        assert result is not None
        return result

    def run_stream(
        self,
        task: str,
        approver: Approver,
        *,
        session_id: str | None = None,
        workspace: Workspace | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Validate the request now and return its event stream.

        Tools see the session's workspace (and the session's memory is used), else
        ``workspace``, else a disposable copy of the template. Session errors
        (SessionNotFoundError / SessionBusyError) are raised here, before any event, so
        the API can map them to HTTP errors. The returned iterator must be consumed by a
        single task: it holds the trace context variable across its yields.
        """
        session = self.sessions.get(session_id) if session_id is not None else None
        if session is not None and session.lock.locked():
            raise SessionBusyError(session_id)
        return self._events(task, approver, session, workspace, request_id)

    async def _events(
        self,
        task: str,
        approver: Approver,
        session: Session | None,
        workspace: Workspace | None,
        request_id: str | None,
    ) -> AsyncIterator[AgentEvent]:
        ephemeral: Workspace | None = None
        if session is not None:
            workspace = session.workspace
        elif workspace is None:
            workspace = ephemeral = self._workspaces.create()
        trace = Trace(trace_id=request_id) if request_id else Trace()
        trace.attributes.update(session_id=session.id if session else None, task_chars=len(task))
        ctx = ToolContext(
            workspace=workspace,
            approver=approver,
            max_output_chars=self._settings.tools.max_output_chars,
            baseline=workspace.snapshot(),
        )
        history = session.history(self._settings.memory.max_turns) if session else []
        try:
            async with session.lock if session else nullcontext():
                with (
                    activate(trace),
                    trace.span("invoke_agent", **{"gen_ai.agent.name": "coding"}),
                ):
                    async for event in self.agent.run_stream(task, ctx, history):
                        if isinstance(event, FinalEvent):
                            event.result.changed_files = changed_files(workspace, ctx.baseline)
                            trace.attributes.update(changed_files=event.result.changed_files)
                        if (
                            isinstance(event, FinalEvent)
                            and session
                            and event.result.status is not RunStatus.ERROR
                        ):
                            session.turns.append(Turn(task, event.result.answer))
                        yield event
        finally:
            if (path := self._trace_sink.write(trace)) is not None:
                logger.info("trace written", extra={"trace": str(path)})
            if ephemeral is not None:
                self._workspaces.destroy(ephemeral)
