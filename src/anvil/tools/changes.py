"""Review changes: a unified diff of the workspace against its state when the run began."""

from __future__ import annotations

import asyncio
import difflib
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from anvil.tools.base import Tool, ToolContext, truncate_middle
from anvil.tools.workspace import Workspace


class ShowChangesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(default=None, description="Limit the diff to this file or directory.")


class ShowChanges(Tool[ShowChangesArgs]):
    name = "show_changes"
    description = (
        "Show a unified diff of every file changed, created or deleted in this task so far. "
        "Use it to review your edits before giving the final answer."
    )
    args_model = ShowChangesArgs

    async def run(self, args: ShowChangesArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: ShowChangesArgs, ctx: ToolContext) -> str:
        prefix = None
        if args.path:
            prefix = ctx.workspace.display(ctx.workspace.resolve(args.path))
        current = ctx.workspace.snapshot()
        diffs: list[str] = []
        for path in changed_paths(ctx.baseline, current):
            if prefix and prefix != "." and not (path == prefix or path.startswith(prefix + "/")):
                continue
            before = ctx.baseline.get(path, "").splitlines(keepends=True)
            after = current.get(path, "").splitlines(keepends=True)
            diffs.append(
                "".join(
                    difflib.unified_diff(
                        before,
                        after,
                        fromfile=f"a/{path}" if path in ctx.baseline else "/dev/null",
                        tofile=f"b/{path}" if path in current else "/dev/null",
                    )
                )
            )
        if not diffs:
            return "No changes since the task started."
        return truncate_middle("\n".join(diffs), ctx.max_output_chars)


def changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """Paths created, deleted or modified between two snapshots, sorted."""
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def changed_files(workspace: Workspace, baseline: Mapping[str, str]) -> list[str]:
    return changed_paths(baseline, workspace.snapshot())
