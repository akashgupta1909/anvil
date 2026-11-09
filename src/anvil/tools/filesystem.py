"""File tools: list, read and write inside the workspace."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from anvil.tools.base import Tool, ToolContext, ToolError, truncate

_HIDDEN = {"__pycache__"}


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListFilesArgs(_Args):
    directory: str = Field(".", description="Directory relative to the workspace root.")


class ReadFileArgs(_Args):
    path: str = Field(description="File path relative to the workspace root.")


class WriteFileArgs(_Args):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="The complete new contents of the file.")


class ListFiles(Tool[ListFilesArgs]):
    name = "list_files"
    description = (
        "List the files and subdirectories of a directory in the workspace, with file sizes. "
        "Use this first to discover what exists before reading files."
    )
    args_model = ListFilesArgs

    async def run(self, args: ListFilesArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: ListFilesArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.directory)
        if not target.is_dir():
            raise ToolError(f'"{args.directory}" is not a directory')
        entries = sorted(p for p in target.iterdir() if p.name not in _HIDDEN)
        lines = [
            f"{ctx.workspace.display(p)}/"
            if p.is_dir()
            else f"{ctx.workspace.display(p)} ({p.stat().st_size} bytes)"
            for p in entries
        ]
        return truncate("\n".join(lines) or "(empty directory)", ctx.max_output_chars)


class ReadFile(Tool[ReadFileArgs]):
    name = "read_file"
    description = (
        "Read a text file from the workspace. Long files are truncated; the result says so."
    )
    args_model = ReadFileArgs

    async def run(self, args: ReadFileArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: ReadFileArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.path)
        if not target.is_file():
            raise ToolError(f'"{args.path}" does not exist or is not a file')
        text = target.read_text(encoding="utf-8", errors="replace")
        return truncate(text, ctx.max_output_chars)


class WriteFile(Tool[WriteFileArgs]):
    name = "write_file"
    description = (
        "Create or overwrite a file in the workspace with the given content (parent "
        "directories are created). This replaces the whole file, so pass the complete "
        "contents. Only use when the user asked for a change."
    )
    side_effects = True
    args_model = WriteFileArgs

    async def run(self, args: WriteFileArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: WriteFileArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.path)
        if target.is_dir():
            raise ToolError(f'"{args.path}" is a directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        return f'Wrote {len(args.content)} characters to "{ctx.workspace.display(target)}"'
