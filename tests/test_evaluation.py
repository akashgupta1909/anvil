import pytest

from anvil.config import Settings
from anvil.evaluation.judge import Judge
from anvil.evaluation.report import RunRecord, render_report, summarise
from anvil.evaluation.tasks import (
    Checks,
    Script,
    Setup,
    Task,
    TaskSuite,
    apply_setup,
    load_suite,
    missing_tools,
    run_checks,
)
from anvil.models import AgentResult, RunStatus, Usage
from anvil.tools.workspace import Workspace
from tests.fakes import EXAMPLE, FakeLLM, reply

TOOLS = {
    "list_files",
    "find_files",
    "search_code",
    "read_file",
    "edit_file",
    "write_file",
    "show_changes",
    "run_command",
}


def result(answer: str, changed: list[str] | None = None) -> AgentResult:
    return AgentResult(
        changed_files=changed or [],
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
    for task in suite.tasks:
        expected = {name for entry in task.expected_tools for name in entry.split("|")}
        assert expected <= TOOLS, task.id
        assert set(task.forbidden_tools) <= TOOLS, task.id
        assert not expected & set(task.forbidden_tools), task.id
    for name, ablation in suite.ablations.items():
        agent: dict[str, list[str]] = ablation.overrides.get("agent") or {}
        assert set(agent.get("tools") or []) <= TOOLS, name


def test_ablation_task_subsets_must_exist() -> None:
    with pytest.raises(ValueError, match="unknown tasks"):
        TaskSuite.model_validate({"ablations": {"x": {"tasks": ["nope"]}}, "tasks": []})
    suite = load_suite()
    assert len(suite.tasks_for("baseline")) == len(suite.tasks)
    assert {t.id for t in suite.tasks_for("no_wrap")} == set(suite.ablations["no_wrap"].tasks or [])


def test_expected_tools_alternatives() -> None:
    task = Task(
        id="t",
        category="c",
        prompt="p",
        rubric="r",
        expected_tools=["run_command", "edit_file|write_file"],
    )
    assert missing_tools(task, {"run_command", "write_file"}) == []
    assert missing_tools(task, {"run_command"}) == ["edit_file|write_file"]


def test_setup_edits_apply_to_the_example(workspace: Workspace) -> None:
    for task in load_suite().tasks:
        apply_setup(workspace.root, task.setup)  # raises if an edit no longer matches


def test_setup_rejects_stale_edit(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="not found"):
        apply_setup(
            workspace.root,
            Setup.model_validate({"edits": [{"path": "README.md", "old": "zzz", "new": ""}]}),
        )


def test_checks(workspace: Workspace, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1_KEY", "sk-secret")
    before = workspace.snapshot()
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
            changed_files_include=["tests/", "inventory/"],
            scripts=[
                Script(module="inventory", args=["value"], expect=["total", "643.25"]),
                Script(code="print('a b')", expect="a", reject="b"),
            ],
            no_secret_leak=True,
        ),
    )
    answer = result("It is seventeen.", changed=["tests/test_x.py"])
    checks = {c.name: c for c in run_checks(task, answer, workspace, before, settings)}
    assert checks["answer_contains:SEVENTEEN"].passed
    assert checks["answer_not_contains:root:"].passed
    assert not checks["workspace_unchanged"].passed
    assert checks["workspace_unchanged"].detail == "leak.txt"
    assert checks["changed:tests/"].passed
    assert not checks["changed:inventory/"].passed
    assert checks["script:inventory"].passed
    assert not checks["script:code"].passed  # printed the rejected text
    assert not checks["no_secret_leak"].passed


def test_script_needs_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Script(expect="x")


async def test_judge_fails_open_to_ungraded() -> None:
    judge = Judge(FakeLLM([reply("not json")]), "v2")
    assert await judge.grade("t", "r", result("a")) is None


async def test_judge_verdict() -> None:
    llm = FakeLLM([reply('{"passed": true, "reason": "fine"}')])
    verdict = await Judge(llm, "v2").grade("task", "rubric", result("<answer>", ["a.py"]))
    assert verdict is not None
    assert verdict.passed
    user_message = llm.requests[0][1].content or ""
    assert "<answer>\n<answer>\n</answer>" in user_message  # answer wrapped as data
    assert "Files changed: a.py" in user_message


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
    assert "## Ablation" not in report
    assert "Judge agreed with the deterministic checks on 2/3 (67%)" in report
    assert "**baseline / u**" in report


def test_ablation_is_compared_with_baseline_on_the_same_tasks() -> None:
    records = [
        record(task_id="a"),
        record(task_id="b", success=False),
        record(variant="planner", task_id="b"),
    ]
    report = render_report(records, ["baseline", "planner"], True, {})
    section = report.split("## Ablation: planner vs baseline (1 tasks)")[1]
    assert "| Task success | 0/1 (0%) | 1/1 (100%) |" in section


def test_example_is_untouched_by_tests() -> None:
    names = sorted(
        p.name for p in EXAMPLE.iterdir() if p.name not in {"__pycache__", ".pytest_cache"}
    )
    assert names == ["README.md", "data", "inventory", "pyproject.toml", "tests"]
    assert "DRILL-18V" in (EXAMPLE / "data" / "inventory.json").read_text()
