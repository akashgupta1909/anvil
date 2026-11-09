"""Aggregate metrics and the markdown report for an evaluation run."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from anvil.config import ModelRef, ModelRole


@dataclass
class RunRecord:
    variant: str
    task_id: str
    category: str
    repeat: int
    request_id: str
    status: str
    success: bool
    checks_passed: bool
    judge_passed: bool | None
    judge_reason: str | None
    tool_correct: bool
    missing_tools: list[str]
    forbidden_called: list[str]
    invalid_calls: int
    tool_calls: int
    steps: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: float
    failed_checks: list[str]
    answer: str


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def summarise(records: Sequence[RunRecord]) -> dict[str, str]:
    n = len(records)
    judged = [r for r in records if r.judge_passed is not None]
    latency = [r.duration_ms / 1000 for r in records]
    return {
        "Runs": str(n),
        "Task success": _pct(sum(r.success for r in records), n),
        # Infrastructure, not agent quality: every endpoint failed (e.g. rate limit/quota).
        "LLM errors": str(sum(r.status == "error" for r in records)),
        "Tool-call correct": _pct(sum(r.tool_correct for r in records), n),
        "Checks pass": _pct(sum(r.checks_passed for r in records), n),
        "Judge pass": _pct(sum(bool(r.judge_passed) for r in judged), len(judged)),
        "Invalid tool calls": str(sum(r.invalid_calls for r in records)),
        "Avg steps": f"{statistics.mean(r.steps for r in records):.1f}",
        "Avg LLM calls": f"{statistics.mean(r.llm_calls for r in records):.1f}",
        "Avg tokens (in/out)": (
            f"{statistics.mean(r.input_tokens for r in records):,.0f} / "
            f"{statistics.mean(r.output_tokens for r in records):,.0f}"
        ),
        "Latency p50 / p95 (s)": f"{statistics.median(latency):.1f} / {_p95(latency):.1f}",
        "Cost / task (list price)": f"${statistics.mean(r.cost_usd for r in records):.5f}",
    }


def render_report(
    records: Sequence[RunRecord],
    variants: Sequence[str],
    use_judge: bool,
    models: dict[ModelRole, list[ModelRef]],
) -> str:
    by_variant = {v: [r for r in records if r.variant == v] for v in variants}
    summaries = {v: summarise(rs) for v, rs in by_variant.items() if rs}
    metrics = list(next(iter(summaries.values())))
    lines = ["# Evaluation report", ""]
    judge_models = ", ".join(m.model for m in models.get("judge", []))
    lines.append(f"Judge: {judge_models if use_judge else 'disabled'}")
    lines += ["", "| Metric | " + " | ".join(summaries) + " |"]
    lines.append("|---" * (len(summaries) + 1) + "|")
    lines += [f"| {m} | " + " | ".join(s[m] for s in summaries.values()) + " |" for m in metrics]

    task_ids = list(dict.fromkeys(r.task_id for r in records))
    lines += ["", "## Per task (successes / runs)", ""]
    lines.append("| Task | " + " | ".join(summaries) + " |")
    lines.append("|---" * (len(summaries) + 1) + "|")
    for task_id in task_ids:
        cells: list[str] = []
        for variant in summaries:
            runs = [r for r in by_variant[variant] if r.task_id == task_id]
            cells.append(f"{sum(r.success for r in runs)}/{len(runs)}")
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")

    judged = [r for r in records if r.judge_passed is not None]
    if judged:
        agree = sum(r.judge_passed == r.checks_passed for r in judged)
        lines += [
            "",
            f"Judge agreed with the deterministic checks on {_pct(agree, len(judged))} runs; "
            f"{len(records) - len(judged)} run(s) were not graded by the judge.",
        ]

    failures = [r for r in records if not r.success]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            reasons = list(r.failed_checks)
            if r.judge_passed is False:
                reasons.append(f"judge: {r.judge_reason}")
            if r.status != "completed":
                reasons.append(f"status={r.status}")
            if not r.tool_correct:
                reasons.append(
                    f"tools: missing={r.missing_tools} forbidden={r.forbidden_called} "
                    f"invalid={r.invalid_calls}"
                )
            lines.append(
                f"- **{r.variant} / {r.task_id}** (trace {r.request_id}): " + "; ".join(reasons)
            )
    return "\n".join(lines) + "\n"
