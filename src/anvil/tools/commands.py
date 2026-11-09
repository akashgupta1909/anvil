"""Run allowlisted programs (no shell) in the workspace."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from anvil.tools.base import Tool, ToolContext, ToolError, truncate_middle

# The child gets a minimal environment: it must not inherit API keys or other secrets,
# since it runs code the model may have written.
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT")
_PYTHON = "python"  # in the allowlist, stands for the interpreter running the agent
_SHELL_TOKENS = {"&&", "||", "|", ";", ">", ">>", "<", "&"}


class RunCommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: list[str] | str = Field(
        description="Program and arguments, preferably as a list, e.g. ['pytest', '-x', "
        "'tests/'] or ['python', '-m', 'inventory', 'report']. It already runs in the "
        "workspace root. No shell: one program per call; no cd, pipes, redirects or &&.",
    )


class RunCommand(Tool[RunCommandArgs]):
    name = "run_command"
    side_effects = True
    args_model = RunCommandArgs

    def __init__(self, allowlist: Mapping[str, Sequence[str]], timeout_seconds: float) -> None:
        """``allowlist`` maps a program name the model may use to the argv prefix that
        is actually run (``python`` in a prefix means this interpreter)."""
        self._allowlist = {
            name: [sys.executable if part == _PYTHON else part for part in prefix]
            for name, prefix in allowlist.items()
        }
        self._timeout = timeout_seconds
        self.description = (
            "Run a program in the workspace root and return its exit code and output. "
            f"Allowed programs: {', '.join(sorted(allowlist))}. Use it to run tests, "
            "scripts and the project's CLI."
        )

    async def run(self, args: RunCommandArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: RunCommandArgs, ctx: ToolContext) -> str:
        program, *rest = _parse(args.command)
        if program == "cd":
            raise ToolError("no cd needed: commands already run in the workspace root")
        prefix = self._allowlist.get(program)
        if prefix is None:
            allowed = ", ".join(sorted(self._allowlist))
            raise ToolError(f"{program!r} is not an allowed program; allowed: {allowed}")
        env = {k: v for k, v in os.environ.items() if k in _ENV_PASSTHROUGH}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [*prefix, *rest],
                cwd=ctx.workspace.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"timed out after {self._timeout:g}s") from exc
        except OSError as exc:
            raise ToolError(f"could not start {program!r}: {exc.strerror or exc}") from exc
        parts = [f"exit code: {proc.returncode}"]
        if proc.stdout:
            parts.append(f"stdout:\n{proc.stdout}")
        if proc.stderr:
            parts.append(f"stderr:\n{proc.stderr}")
        if not proc.stdout and not proc.stderr:
            parts.append("(no output)")
        return truncate_middle("\n".join(parts), ctx.max_output_chars)


def _parse(command: list[str] | str) -> list[str]:
    """Accept a list or a plain string (split like a shell would), but reject shell
    syntax with a message that tells the model what to do instead."""
    try:
        argv = shlex.split(command) if isinstance(command, str) else list(command)
    except ValueError as exc:
        raise ToolError(f"could not parse command: {exc}") from exc
    if not argv:
        raise ToolError("command is empty")
    if _SHELL_TOKENS.intersection(argv):
        raise ToolError(
            "shell syntax is not supported (no &&, |, ;, redirects): run one program per call"
        )
    return argv
