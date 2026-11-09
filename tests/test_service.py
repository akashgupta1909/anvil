"""Application service and session memory."""

import json
from pathlib import Path

import pytest

from anvil.config import Settings
from anvil.container import build_container
from anvil.memory.sessions import InMemorySessionStore, SessionNotFoundError
from anvil.service import AgentService, SessionBusyError
from anvil.tools.approval import AllowAll
from anvil.tools.workspace import WorkspaceManager
from tests.fakes import EXAMPLE, FakeLLM, reply


def service(settings: Settings, llm: FakeLLM) -> AgentService:
    return build_container(settings, {"main": llm, "fast": llm, "judge": llm}).service


async def test_one_shot_uses_and_removes_a_disposable_workspace(settings: Settings) -> None:
    llm = FakeLLM([reply(None, ("write_file", '{"path": "x.txt", "content": "1"}')), reply("ok")])
    result = await service(settings, llm).run("write", AllowAll())
    assert result.tool_calls[0].ok
    assert list(settings.workspace.root.iterdir()) == []  # copy was destroyed
    assert (EXAMPLE / "x.txt").exists() is False  # template untouched


async def test_trace_is_written_with_request_id(settings: Settings) -> None:
    await service(settings, FakeLLM([reply("ok")])).run("hi", AllowAll(), request_id="req-1")
    trace = json.loads((settings.tracing.dir / "req-1.json").read_text())
    assert trace["attributes"]["status"] == "completed"
    assert [s["name"] for s in trace["spans"]] == ["invoke_agent"]
    assert trace["attributes"]["system_prompt"].startswith("system@v1#")


async def test_session_keeps_workspace_and_memory(settings: Settings) -> None:
    llm = FakeLLM(
        [
            reply(None, ("write_file", '{"path": "note.txt", "content": "1"}')),
            reply("wrote it"),
            reply("you asked me to write a note"),
        ]
    )
    svc = service(settings, llm)
    session = svc.sessions.create()
    await svc.run("write a note", AllowAll(), session_id=session.id)
    await svc.run("what did I ask?", AllowAll(), session_id=session.id)
    assert (session.workspace.root / "note.txt").read_text() == "1"
    second_request = [(m.role, m.content) for m in llm.requests[-1][1:]]
    assert second_request == [
        ("user", "write a note"),
        ("assistant", "wrote it"),
        ("user", "what did I ask?"),
    ]


async def test_concurrent_runs_in_one_session_are_rejected(settings: Settings) -> None:
    svc = service(settings, FakeLLM([reply("ok")]))
    session = svc.sessions.create()
    await session.lock.acquire()
    with pytest.raises(SessionBusyError):
        await svc.run("hi", AllowAll(), session_id=session.id)


async def test_unknown_session(settings: Settings) -> None:
    with pytest.raises(SessionNotFoundError):
        await service(settings, FakeLLM([])).run("hi", AllowAll(), session_id="nope")


def test_store_evicts_least_recently_used(tmp_path: Path) -> None:
    store = InMemorySessionStore(
        WorkspaceManager(EXAMPLE, tmp_path), max_sessions=2, ttl_seconds=60
    )
    a, b = store.create(), store.create()
    store.get(a.id)  # a is now most recent
    store.create()
    with pytest.raises(SessionNotFoundError):
        store.get(b.id)
    assert not b.workspace.root.exists()
    assert store.get(a.id) is a


def test_history_keeps_last_turns(tmp_path: Path) -> None:
    from anvil.memory.sessions import Turn

    store = InMemorySessionStore(
        WorkspaceManager(EXAMPLE, tmp_path), max_sessions=2, ttl_seconds=60
    )
    session = store.create()
    session.turns = [Turn(f"u{i}", f"a{i}") for i in range(5)]
    assert [m.content for m in session.history(max_turns=2)] == ["u3", "a3", "u4", "a4"]
