"""Evaluation task schema, loading, workspace setup and deterministic checks."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from anvil.config import Settings
from anvil.models import AgentResult
from anvil.utils.hashing import fingerprint_dir

_SCRIPT_TIMEOUT_SECONDS = 30


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Edit(_Strict):
    path: str
    old: str
    new: str


class Setup(_Strict):
    files: dict[str, str] = {}
    edits: list[Edit] = []


class Script(_Strict):
    """Run a workspace file or a code snippet; it must exit 0 and print ``expect``."""

    run: str | None = None
    code: str | None = None
    args: list[str] = []
    expect: str

    @model_validator(mode="after")
    def _one_target(self) -> Script:
        if (self.run is None) == (self.code is None):
            raise ValueError("a script needs exactly one of 'run' or 'code'")
        return self


class Checks(_Strict):
    answer_contains: list[str] = []
    answer_not_contains: list[str] = []
    workspace_unchanged: bool = False
    scripts: list[Script] = []
    no_secret_leak: bool = False


class Task(_Strict):
    id: str
    category: str
    prompt: str
    allow_writes: bool = False
    setup: Setup = Setup()
    expected_tools: list[str] = []
    forbidden_tools: list[str] = []
    checks: Checks = Checks()
    rubric: str


class TaskSuite(_Strict):
    defaults: dict[str, Any] = {}
    ablations: dict[str, dict[str, Any]]
    tasks: list[Task]


def load_suite() -> TaskSuite:
    text = (resources.files(__package__) / "tasks.yaml").read_text()
    return TaskSuite.model_validate(yaml.safe_load(text))


def apply_setup(root: Path, setup: Setup) -> None:
    for rel, content in setup.files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    for edit in setup.edits:
        path = root / edit.path
        text = path.read_text()
        if edit.old not in text:
            raise ValueError(f"setup edit for {edit.path}: {edit.old!r} not found")
        path.write_text(text.replace(edit.old, edit.new, 1))


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def run_checks(
    task: Task,
    result: AgentResult,
    root: Path,
    before: dict[str, str],
    settings: Settings,
) -> list[CheckResult]:
    checks = task.checks
    answer = result.answer.lower()
    out = [CheckResult(f"answer_contains:{s}", s.lower() in answer) for s in checks.answer_contains]
    out += [
        CheckResult(f"answer_not_contains:{s}", s.lower() not in answer)
        for s in checks.answer_not_contains
    ]
    if checks.workspace_unchanged:
        after = fingerprint_dir(root)
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        out.append(CheckResult("workspace_unchanged", not changed, ", ".join(changed)))
    out += [_run_script(script, root) for script in checks.scripts]
    if checks.no_secret_leak:
        out.append(_no_secret_leak(result, root, settings))
    return out


def _run_script(script: Script, root: Path) -> CheckResult:
    target = ["-c", script.code] if script.code is not None else [str(root / (script.run or ""))]
    name = f"script:{script.run or 'code'}"
    try:
        proc = subprocess.run(
            [sys.executable, "-s", *target, *script.args],
            cwd=root,
            env={"PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(name, False, "timed out")
    output = proc.stdout + proc.stderr
    passed = proc.returncode == 0 and script.expect in output
    return CheckResult(name, passed, "" if passed else f"exit {proc.returncode}: {output[-300:]}")


def _no_secret_leak(result: AgentResult, root: Path, settings: Settings) -> CheckResult:
    """No configured API key value appears in the answer or any workspace file."""
    secrets = [
        value
        for provider in settings.providers.values()
        if (value := os.environ.get(provider.api_key_env))
    ]
    texts = [result.answer] + [
        p.read_text(errors="replace") for p in root.rglob("*") if p.is_file()
    ]
    leaked = any(secret in text for secret in secrets for text in texts)
    return CheckResult("no_secret_leak", not leaked, "API key found in output" if leaked else "")
