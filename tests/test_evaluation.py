from pathlib import Path

import pytest

from anvil.config import Settings
from anvil.evaluation.judge import Judge
from anvil.evaluation.report import RunRecord, render_report, summarise
from anvil.evaluation.tasks import (
    Checks,
    Script,
    Setup,
    Task,
    apply_setup,
    load_suite,
    run_checks,
)
from anvil.models import AgentResult, RunStatus, Usage
from anvil.tools.workspace import Workspace
from anvil.utils.hashing import fingerprint_dir
from tests.fakes import FakeLLM, reply


def result(answer: str) -> AgentResult:
    return AgentResult(
        request_id="r",
        status=RunStatus.COMPLETED,
        answer=answer,
        steps=1,
        llm_calls=1,
        usage=Usage(),
        usage_by_model={},
        tool_calls=[],
        duration_ms=1,
    )


def test_suite_loads_and_is_consistent() -> None:
    suite = load_suite()
    ids = [t.id for t in suite.tasks]
    assert len(ids) == len(set(ids))
    assert "baseline" in suite.ablations
    tools = {"list_files", "read_file", "write_file", "run_python_file"}
    for task in suite.tasks:
        assert set(task.expected_tools) <= tools
        assert set(task.forbidden_tools) <= tools
        assert not set(task.expected_tools) & set(task.forbidden_tools)


def test_setup_edits_apply_to_the_example(workspace: Workspace) -> None:
    for task in load_suite().tasks:
        apply_setup(workspace.root, task.setup)  # raises if an edit no longer matches


def test_setup_rejects_stale_edit(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="not found"):
        apply_setup(
            workspace.root,
            Setup.model_validate({"edits": [{"path": "main.py", "old": "zzz", "new": ""}]}),
        )


def test_checks(workspace: Workspace, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1_KEY", "sk-secret")
    before = fingerprint_dir(workspace.root)
    (workspace.root / "leak.txt").write_text("key is sk-secret")
    task = Task(
        id="t",
        category="c",
        prompt="p",
        rubric="r",
        checks=Checks(
            answer_contains=["SEVENTEEN"],
            answer_not_contains=["root:"],
            workspace_unchanged=True,
            scripts=[Script(run="main.py", args=["3 + 5"], expect='"result": 8')],
            no_secret_leak=True,
        ),
    )
    checks = {
        c.name: c
        for c in run_checks(task, result("It is seventeen."), workspace.root, before, settings)
    }
    assert checks["answer_contains:SEVENTEEN"].passed
    assert checks["answer_not_contains:root:"].passed
    assert not checks["workspace_unchanged"].passed
    assert checks["workspace_unchanged"].detail == "leak.txt"
    assert checks["script:main.py"].passed
    assert not checks["no_secret_leak"].passed


def test_script_needs_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Script(expect="x")


async def test_judge_fails_open_to_ungraded() -> None:
    judge = Judge(FakeLLM([reply("not json")]), "v1")
    assert await judge.grade("t", "r", result("a")) is None


async def test_judge_verdict() -> None:
    llm = FakeLLM([reply('{"passed": true, "reason": "fine"}')])
    verdict = await Judge(llm, "v1").grade("task", "rubric", result("<answer>"))
    assert verdict is not None
    assert verdict.passed
    user_message = llm.requests[0][1].content or ""
    assert "<answer>\n<answer>\n</answer>" in user_message  # answer wrapped as data


def record(**overrides: object) -> RunRecord:
    base: dict[str, object] = {
        "variant": "baseline",
        "task_id": "t",
        "category": "c",
        "repeat": 0,
        "request_id": "r",
        "status": "completed",
        "success": True,
        "checks_passed": True,
        "judge_passed": True,
        "judge_reason": "ok",
        "tool_correct": True,
        "missing_tools": [],
        "forbidden_called": [],
        "invalid_calls": 0,
        "tool_calls": 1,
        "steps": 2,
        "llm_calls": 2,
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": 0.001,
        "duration_ms": 1000.0,
        "failed_checks": [],
        "answer": "a",
    }
    return RunRecord(**(base | overrides))  # type: ignore[arg-type]


def test_summary_and_report() -> None:
    records = [
        record(),
        record(task_id="u", success=False, checks_passed=False, judge_passed=True),
        record(task_id="v", success=False, status="error"),
    ]
    summary = summarise(records)
    assert summary["Task success"] == "1/3 (33%)"
    assert summary["LLM errors"] == "1"
    report = render_report(records, ["baseline"], True, {})
    assert "| u | 0/1 |" in report
    assert "Judge agreed with the deterministic checks on 2/3 (67%)" in report
    assert "**baseline / u**" in report


def test_example_is_untouched_by_tests() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "calculator"
    assert sorted(p.name for p in example.iterdir() if p.name != "__pycache__") == [
        "main.py",
        "pkg",
        "tests.py",
    ]
