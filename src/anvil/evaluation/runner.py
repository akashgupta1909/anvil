"""Runs the task suite for one or more ablation variants and writes a report.

Results are appended to ``results.jsonl`` as each run finishes. ``resume`` continues an
earlier run directory: completed runs are kept, runs that ended in an LLM error are
retried. The runner stops early after consecutive LLM errors (e.g. a daily token quota
is exhausted) so the run can be resumed later instead of wasting time on retries.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from anvil.config import Settings, deep_merge
from anvil.container import Container, build_container
from anvil.evaluation.judge import Judge
from anvil.evaluation.report import RunRecord, render_report
from anvil.evaluation.tasks import (
    CheckResult,
    Task,
    apply_setup,
    load_suite,
    missing_tools,
    run_checks,
)
from anvil.models import AgentResult
from anvil.tools.approval import AllowAll, DenyAll

MODEL_MISTAKES = {"unknown_tool", "bad_arguments"}
_MAX_CONSECUTIVE_LLM_ERRORS = 2


async def run_evaluation(
    settings: Settings,
    *,
    ablations: list[str] | None = None,
    task_ids: list[str] | None = None,
    repeats: int = 1,
    use_judge: bool = True,
    resume: Path | None = None,
    out_root: Path = Path("logs/eval"),
) -> int:
    suite = load_suite()
    variants = ablations or ["baseline"]
    unknown = [v for v in variants if v not in suite.ablations]
    if unknown:
        print(f"Unknown ablation(s): {unknown}; have {list(suite.ablations)}", file=sys.stderr)
        return 2

    out_dir = resume or out_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    records = _load_records(results_path)
    done = {(r.variant, r.task_id, r.repeat) for r in records if r.status != "error"}
    records = [r for r in records if r.status != "error"]
    results_path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))

    consecutive_errors = 0
    stopped = False
    with results_path.open("a") as sink:
        for variant in variants:
            tasks = [t for t in suite.tasks_for(variant) if not task_ids or t.id in task_ids]
            overrides = deep_merge(suite.defaults, suite.ablations[variant].overrides)
            variant_settings = Settings(**deep_merge(settings.model_dump(), overrides))
            container = build_container(variant_settings)
            judge = (
                Judge(container.llms["judge"], variant_settings.agent.prompts.judge)
                if use_judge
                else None
            )
            for repeat in range(repeats):
                for task in tasks:
                    if (variant, task.id, repeat) in done:
                        continue
                    record = await _run_task(container, judge, task, variant, repeat)
                    records.append(record)
                    sink.write(json.dumps(asdict(record)) + "\n")
                    sink.flush()
                    _print_progress(record)
                    consecutive_errors = consecutive_errors + 1 if record.status == "error" else 0
                    if consecutive_errors >= _MAX_CONSECUTIVE_LLM_ERRORS:
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break

    report = render_report(records, list(suite.ablations), use_judge, settings.models)
    (out_dir / "report.md").write_text(report)
    print(report)
    if stopped:
        print(
            f"\nStopped after {_MAX_CONSECUTIVE_LLM_ERRORS} consecutive LLM errors (quota?). "
            f"Continue later with: anvil eval --resume {out_dir} "
            + " ".join(f"--ablation {v}" for v in variants),
            file=sys.stderr,
        )
        return 1
    print(f"\nResults: {out_dir}", file=sys.stderr)
    return 0


def _load_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    return [RunRecord(**json.loads(line)) for line in path.read_text().splitlines() if line]


def _print_progress(record: RunRecord) -> None:
    mark = "PASS" if record.success else ("ERROR" if record.status == "error" else "FAIL")
    print(
        f"[{record.variant}] {record.task_id:<26} {mark:<5} steps={record.steps} "
        f"tools={record.tool_calls} {record.duration_ms / 1000:.1f}s"
        + (f"  failed={record.failed_checks}" if record.failed_checks else ""),
        file=sys.stderr,
    )


async def _run_task(
    container: Container, judge: Judge | None, task: Task, variant: str, repeat: int
) -> RunRecord:
    settings = container.settings
    workspace = container.workspaces.create()
    try:
        apply_setup(workspace.root, task.setup)
        before = workspace.snapshot()
        approver = AllowAll() if task.allow_writes else DenyAll()
        result = await container.service.run(task.prompt, approver, workspace=workspace)
        checks = run_checks(task, result, workspace, before, settings)
    finally:
        container.workspaces.destroy(workspace)

    verdict = await judge.grade(task.prompt, task.rubric, result) if judge else None
    called = {c.name for c in result.tool_calls}
    missing = missing_tools(task, called)
    forbidden = sorted(set(task.forbidden_tools) & called)
    invalid = sum(1 for c in result.tool_calls if c.error_type in MODEL_MISTAKES)
    checks_passed = all(c.passed for c in checks)
    judge_passed = verdict.passed if verdict else None
    return RunRecord(
        variant=variant,
        task_id=task.id,
        category=task.category,
        repeat=repeat,
        request_id=result.request_id,
        status=result.status.value,
        success=result.status.value == "completed" and checks_passed and judge_passed is not False,
        checks_passed=checks_passed,
        judge_passed=judge_passed,
        judge_reason=verdict.reason if verdict else None,
        tool_correct=not missing and not forbidden and invalid == 0,
        missing_tools=missing,
        forbidden_called=forbidden,
        invalid_calls=invalid,
        tool_calls=len(result.tool_calls),
        steps=result.steps,
        llm_calls=result.llm_calls,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=_cost(result, settings),
        duration_ms=result.duration_ms,
        failed_checks=[_describe(c) for c in checks if not c.passed],
        answer=result.answer,
    )


def _cost(result: AgentResult, settings: Settings) -> float:
    total = 0.0
    for model, usage in result.usage_by_model.items():
        price = settings.pricing.get(model)
        if price:
            total += (usage.input_tokens * price.input + usage.output_tokens * price.output) / 1e6
    return total


def _describe(check: CheckResult) -> str:
    return f"{check.name} ({check.detail})" if check.detail else check.name
