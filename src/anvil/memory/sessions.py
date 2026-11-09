"""Conversation memory: sessions that persist a workspace and prior turns.

A turn is stored as (user request, final answer) only. Raw tool traffic is dropped:
it is large (the free tier allows 8k tokens/min) and it is untrusted, so injected text
in a file cannot persist across turns. The ``SessionStore`` protocol is the seam for a
shared backend (e.g. Redis) when running more than one API process.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

from anvil.models import Message
from anvil.tools.workspace import Workspace, WorkspaceManager


class SessionNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass
class Session:
    id: str
    workspace: Workspace
    owns_workspace: bool  # True if the store created it and must delete it
    turns: list[Turn] = field(default_factory=lambda: [])
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)

    def history(self, max_turns: int) -> list[Message]:
        messages: list[Message] = []
        for turn in self.turns[-max_turns:]:
            messages.append(Message(role="user", content=turn.user))
            messages.append(Message(role="assistant", content=turn.assistant))
        return messages


class SessionStore(Protocol):
    def create(self, workspace: Workspace | None = None) -> Session: ...
    def get(self, session_id: str) -> Session: ...
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Process-local store with LRU eviction and idle expiry."""

    def __init__(self, workspaces: WorkspaceManager, max_sessions: int, ttl_seconds: int) -> None:
        self._workspaces = workspaces
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def create(self, workspace: Workspace | None = None) -> Session:
        self._expire()
        while len(self._sessions) >= self._max_sessions:
            self._drop(next(iter(self._sessions)))
        session = Session(
            id=uuid.uuid4().hex,
            workspace=workspace or self._workspaces.create(),
            owns_workspace=workspace is None,
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        self._expire()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        session.last_used = time.monotonic()
        self._sessions.move_to_end(session_id)
        return session

    def delete(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        self._drop(session_id)

    def _expire(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for session_id in [s.id for s in self._sessions.values() if s.last_used < cutoff]:
            if not self._sessions[session_id].lock.locked():
                self._drop(session_id)

    def _drop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id)
        if session.owns_workspace:
            self._workspaces.destroy(session.workspace)
