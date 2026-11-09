import json

import pytest
from fastapi.testclient import TestClient

from anvil.api.app import create_app
from anvil.config import Settings
from anvil.container import build_container
from anvil.llm.client import LLMUnavailableError
from tests.fakes import FakeLLM, reply

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANVIL_API_KEY", "test-key")


def client(settings: Settings, llm: FakeLLM) -> TestClient:
    container = build_container(settings, {"main": llm, "fast": llm, "judge": llm})
    return TestClient(create_app(container))


def test_health_is_public_and_has_request_id(settings: Settings) -> None:
    response = client(settings, FakeLLM([])).get("/health", headers={"x-request-id": "abc"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"] == "abc"


def test_runs_require_auth(settings: Settings) -> None:
    response = client(settings, FakeLLM([])).post("/v1/runs", json={"task": "hi"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.json()["error"]["request_id"]


def test_refuses_to_start_without_key(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANVIL_API_KEY")
    with pytest.raises(RuntimeError, match="ANVIL_API_KEY"):
        client(settings, FakeLLM([]))


def test_run_returns_result(settings: Settings) -> None:
    response = client(settings, FakeLLM([reply("hello")])).post(
        "/v1/runs", json={"task": "hi"}, headers=AUTH | {"x-request-id": "r1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "hello"
    assert body["status"] == "completed"
    assert body["request_id"] == "r1"


def test_writes_are_denied_unless_allowed(settings: Settings) -> None:
    script = [reply(None, ("write_file", '{"path": "a.txt", "content": "1"}')), reply("done")]
    body = (
        client(settings, FakeLLM(list(script)))
        .post("/v1/runs", json={"task": "w"}, headers=AUTH)
        .json()
    )
    assert body["tool_calls"][0]["error_type"] == "denied"
    body = (
        client(settings, FakeLLM(list(script)))
        .post("/v1/runs", json={"task": "w", "allow_writes": True}, headers=AUTH)
        .json()
    )
    assert body["tool_calls"][0]["ok"] is True


def test_validation_errors_are_mapped(settings: Settings) -> None:
    response = client(settings, FakeLLM([])).post("/v1/runs", json={"task": ""}, headers=AUTH)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_llm_outage_maps_to_503(settings: Settings) -> None:
    response = client(settings, FakeLLM([LLMUnavailableError("down")])).post(
        "/v1/runs", json={"task": "hi"}, headers=AUTH
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


def test_unknown_session_is_404(settings: Settings) -> None:
    response = client(settings, FakeLLM([])).post(
        "/v1/runs/stream", json={"task": "hi", "session_id": "nope"}, headers=AUTH
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_session_lifecycle(settings: Settings) -> None:
    api = client(settings, FakeLLM([reply("one"), reply("two")]))
    session_id = api.post("/v1/sessions", headers=AUTH).json()["session_id"]
    for task in ("first", "second"):
        response = api.post("/v1/runs", json={"task": task, "session_id": session_id}, headers=AUTH)
        assert response.status_code == 200
    assert api.delete(f"/v1/sessions/{session_id}", headers=AUTH).status_code == 204
    assert api.delete(f"/v1/sessions/{session_id}", headers=AUTH).status_code == 404


def test_stream_emits_events_then_final(settings: Settings) -> None:
    llm = FakeLLM([reply(None, ("list_files", "{}")), reply("all done")])
    with client(settings, llm).stream(
        "POST", "/v1/runs/stream", json={"task": "ls"}, headers=AUTH
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            line.removeprefix("event: ").strip()
            for line in response.iter_lines()
            if line.startswith("event:")
        ]
        assert events == ["tool_call", "tool_result", "text_delta", "text_delta", "final"]


def test_stream_final_event_payload(settings: Settings) -> None:
    with client(settings, FakeLLM([reply("hi")])).stream(
        "POST", "/v1/runs/stream", json={"task": "x"}, headers=AUTH
    ) as response:
        data = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data:")
        ]
    final = json.loads(data[-1])
    assert final["type"] == "final"
    assert final["result"]["answer"] == "hi"
