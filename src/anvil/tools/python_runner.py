"""Run a Python file from the workspace in a subprocess."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from anvil.tools.base import Tool, ToolContext, ToolError, truncate

# The child gets a minimal environment: it must not inherit API keys or other secrets,
# since it runs code the model may have written.
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT")


class RunPythonArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Python file relative to the workspace root, e.g. 'main.py'.")
    args: list[str] = Field(
        default_factory=list, description="Command-line arguments passed to the script."
    )


class RunPythonFile(Tool[RunPythonArgs]):
    name = "run_python_file"
    description = (
        "Run a Python file from the workspace (working directory = workspace root) and "
        "return its exit code, stdout and stderr. Use it to run programs and test files. "
        "Each argument is passed separately, e.g. args=['3 + 5']."
    )
    side_effects = True
    args_model = RunPythonArgs

    async def run(self, args: RunPythonArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: RunPythonArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.path)
        if target.suffix != ".py":
            raise ToolError(f'"{args.path}" is not a Python (.py) file')
        if not target.is_file():
            raise ToolError(f'"{args.path}" does not exist')
        env = {k: v for k, v in os.environ.items() if k in _ENV_PASSTHROUGH}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-s", str(target), *args.args],
                cwd=ctx.workspace.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=ctx.run_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"timed out after {ctx.run_timeout_seconds:g}s") from exc
        parts = [f"exit code: {proc.returncode}"]
        if proc.stdout:
            parts.append(f"stdout:\n{proc.stdout}")
        if proc.stderr:
            parts.append(f"stderr:\n{proc.stderr}")
        if not proc.stdout and not proc.stderr:
            parts.append("(no output)")
        return truncate("\n".join(parts), ctx.max_output_chars)
