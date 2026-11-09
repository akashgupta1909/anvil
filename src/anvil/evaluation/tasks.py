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
from anvil.tools.workspace import Workspace

_SCRIPT_TIMEOUT_SECONDS = 60


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
    """Run a workspace file, a code snippet or a module (``python -m``) in the workspace.
    It must exit 0, print every ``expect`` string and none of ``reject``."""

    run: str | None = None
    code: str | None = None
    module: str | None = None
    args: list[str] = []
    expect: str | list[str]
    reject: str | None = None

    @model_validator(mode="after")
    def _one_target(self) -> Script:
        if sum(x is not None for x in (self.run, self.code, self.module)) != 1:
            raise ValueError("a script needs exactly one of 'run', 'code' or 'module'")
        return self

    @property
    def name(self) -> str:
        return f"script:{self.run or self.module or 'code'}"


class Checks(_Strict):
    answer_contains: list[str] = []
    answer_not_contains: list[str] = []
    workspace_unchanged: bool = False
    changed_files_include: list[str] = []  # path prefixes the agent must have changed
    scripts: list[Script] = []
    no_secret_leak: bool = False


class Task(_Strict):
    id: str
    category: str
    prompt: str
    allow_writes: bool = False
    setup: Setup = Setup()
    expected_tools: list[str] = []  # each entry: a tool name, or alternatives "a|b"
    forbidden_tools: list[str] = []
    checks: Checks = Checks()
    rubric: str


class Ablation(_Strict):
    overrides: dict[str, Any] = {}
    tasks: list[str] | None = None  # None = every task


class TaskSuite(_Strict):
    defaults: dict[str, Any] = {}
    ablations: dict[str, Ablation]
    tasks: list[Task]

    @model_validator(mode="after")
    def _known_task_ids(self) -> TaskSuite:
        ids = {t.id for t in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("duplicate task ids")
        for name, ablation in self.ablations.items():
            unknown = set(ablation.tasks or []) - ids
            if unknown:
                raise ValueError(f"ablation {name!r}: unknown tasks {sorted(unknown)}")
        return self

    def tasks_for(self, ablation: str) -> list[Task]:
        wanted = self.ablations[ablation].tasks
        return [t for t in self.tasks if wanted is None or t.id in wanted]


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


def missing_tools(task: Task, called: set[str]) -> list[str]:
    """Expected entries not satisfied by the calls made ("a|b" needs a or b)."""
    return [e for e in task.expected_tools if not set(e.split("|")) & called]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def run_checks(
    task: Task,
    result: AgentResult,
    workspace: Workspace,
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
    after = workspace.snapshot()
    if checks.workspace_unchanged:
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        out.append(CheckResult("workspace_unchanged", not changed, ", ".join(changed)))
    for prefix in checks.changed_files_include:
        hit = any(path.startswith(prefix) for path in result.changed_files)
        out.append(CheckResult(f"changed:{prefix}", hit, "" if hit else "no file changed there"))
    out += [_run_script(script, workspace.root) for script in checks.scripts]
    if checks.no_secret_leak:
        out.append(_no_secret_leak(result, after, settings))
    return out


def _run_script(script: Script, root: Path) -> CheckResult:
    if script.code is not None:
        target = ["-c", script.code]
    elif script.module is not None:
        target = ["-m", script.module]
    else:
        target = [str(root / (script.run or ""))]
    try:
        proc = subprocess.run(
            [sys.executable, "-s", *target, *script.args],
            cwd=root,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(script.name, False, "timed out")
    output = proc.stdout + proc.stderr
    expected = [script.expect] if isinstance(script.expect, str) else script.expect
    passed = (
        proc.returncode == 0
        and all(e in output for e in expected)
        and not (script.reject and script.reject in output)
    )
    detail = "" if passed else f"exit {proc.returncode}: {output[-300:]}"
    return CheckResult(script.name, passed, detail)


def _no_secret_leak(result: AgentResult, files: dict[str, str], settings: Settings) -> CheckResult:
    """No configured API key value appears in the answer or any workspace file."""
    secrets = [
        value
        for provider in settings.providers.values()
        if (value := os.environ.get(provider.api_key_env))
    ]
    texts = [result.answer, *files.values()]
    leaked = any(secret in text for secret in secrets for text in texts)
    return CheckResult("no_secret_leak", not leaked, "API key found in output" if leaked else "")
