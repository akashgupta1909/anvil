"""Search tools: find files by glob pattern and search file contents by regex."""

from __future__ import annotations

import asyncio
import fnmatch
import re

from pydantic import BaseModel, ConfigDict, Field

from anvil.tools.base import Tool, ToolContext, ToolError
from anvil.tools.workspace import read_text_file

_MAX_FILES = 200
_MAX_MATCHES = 100
_MAX_LINE_CHARS = 200
_MAX_SEARCH_FILE_BYTES = 1_000_000


class FindFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        description="Glob pattern matched against paths relative to the workspace root, "
        "e.g. '**/*.py', 'tests/test_*.py', '**/config*'."
    )


class SearchCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="Python regular expression, e.g. 'def \\w+_total'.")
    path: str = Field(default=".", description="File or directory to search, relative to the root.")
    file_glob: str | None = Field(
        default=None, description="Only search files matching, e.g. '*.py'."
    )
    ignore_case: bool = False


class FindFiles(Tool[FindFilesArgs]):
    name = "find_files"
    description = (
        "Find files anywhere in the workspace whose path matches a glob pattern. "
        "Returns matching paths, sorted."
    )
    args_model = FindFilesArgs

    async def run(self, args: FindFilesArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: FindFilesArgs, ctx: ToolContext) -> str:
        pattern = args.pattern.removeprefix("./")
        matches = [
            rel
            for path in ctx.workspace.iter_files()
            if _glob_match(rel := ctx.workspace.display(path), pattern)
        ]
        if not matches:
            return f"No files match {args.pattern!r}."
        shown = matches[:_MAX_FILES]
        more = f"\n[... {len(matches) - len(shown)} more]" if len(matches) > len(shown) else ""
        return "\n".join(shown) + more


class SearchCode(Tool[SearchCodeArgs]):
    name = "search_code"
    description = (
        "Search file contents in the workspace with a regular expression. Returns matching "
        "lines as 'path:line: text'. Use it to find definitions, usages and error messages "
        "before reading files."
    )
    args_model = SearchCodeArgs

    async def run(self, args: SearchCodeArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: SearchCodeArgs, ctx: ToolContext) -> str:
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        start = ctx.workspace.resolve(args.path)
        if not start.exists():
            raise ToolError(f'"{args.path}" does not exist')
        results: list[str] = []
        total = 0
        for path in ctx.workspace.iter_files(start):
            if args.file_glob and not fnmatch.fnmatch(path.name, args.file_glob):
                continue
            text = read_text_file(path, _MAX_SEARCH_FILE_BYTES)
            if text is None:
                continue
            rel = ctx.workspace.display(path)
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    total += 1
                    if len(results) < _MAX_MATCHES:
                        results.append(f"{rel}:{number}: {line.strip()[:_MAX_LINE_CHARS]}")
        if not results:
            return f"No matches for {args.pattern!r}."
        more = f"\n[... {total - len(results)} more matches]" if total > len(results) else ""
        return "\n".join(results) + more


def _glob_match(path: str, pattern: str) -> bool:
    """fnmatch, plus '**/' also matching zero directories ('**/*.py' matches 'a.py')."""
    if fnmatch.fnmatch(path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:])
