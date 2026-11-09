"""File tools: list, read (with line ranges), edit (exact replace) and write."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from anvil.tools.base import Tool, ToolContext, ToolError, truncate
from anvil.tools.workspace import IGNORED_DIRS

_SNIPPET_CONTEXT_LINES = 3


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListFilesArgs(_Args):
    directory: str = Field(default=".", description="Directory relative to the workspace root.")


class ReadFileArgs(_Args):
    path: str = Field(description="File path relative to the workspace root.")
    start_line: int = Field(default=1, ge=1, description="First line to read (1-based).")
    end_line: int | None = Field(
        default=None, ge=1, description="Last line to read (inclusive). Omit to read to the end."
    )


class EditFileArgs(_Args):
    path: str = Field(description="File path relative to the workspace root.")
    old_text: str = Field(
        min_length=1,
        description="Exact text to replace, copied from the file without line-number prefixes. "
        "Include enough surrounding lines to match exactly one place.",
    )
    new_text: str = Field(description="Replacement text (may be empty to delete).")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of one.")


class WriteFileArgs(_Args):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="The complete contents of the file.")


class ListFiles(Tool[ListFilesArgs]):
    name = "list_files"
    description = (
        "List the files and subdirectories of one directory in the workspace, with file "
        "sizes. Use find_files to list files recursively by pattern."
    )
    args_model = ListFilesArgs

    async def run(self, args: ListFilesArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: ListFilesArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.directory)
        if not target.is_dir():
            raise ToolError(f'"{args.directory}" is not a directory')
        entries = sorted(p for p in target.iterdir() if p.name not in IGNORED_DIRS)
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
        "Read a text file from the workspace. Each line is prefixed with its line number "
        "and a tab (the prefix is not part of the file). Long files are cut off; read the "
        "rest with start_line/end_line."
    )
    args_model = ReadFileArgs

    async def run(self, args: ReadFileArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: ReadFileArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.path)
        if not target.is_file():
            raise ToolError(f'"{args.path}" does not exist or is not a file')
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return "(empty file)"
        first = args.start_line
        last = min(args.end_line or len(lines), len(lines))
        if first > len(lines):
            raise ToolError(f"start_line {first} is past the end of the file ({len(lines)} lines)")
        out: list[str] = []
        used = 0
        for number in range(first, last + 1):
            line = f"{number:>5}\t{lines[number - 1]}"
            if used + len(line) > ctx.max_output_chars and out:
                out.append(
                    f"[... cut off at line {number - 1} of {len(lines)}; "
                    f"continue with start_line={number}]"
                )
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out)


class EditFile(Tool[EditFileArgs]):
    name = "edit_file"
    description = (
        "Change part of an existing file by replacing exact text. old_text must match the "
        "file exactly (whitespace included, no line-number prefixes) and, unless replace_all "
        "is set, exactly one place. Prefer this over write_file for existing files."
    )
    side_effects = True
    args_model = EditFileArgs

    async def run(self, args: EditFileArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: EditFileArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.path)
        if not target.is_file():
            raise ToolError(f'"{args.path}" does not exist; use write_file to create it')
        text = target.read_text(encoding="utf-8")
        count = text.count(args.old_text)
        if count == 0:
            raise ToolError(
                "old_text was not found. Read the file again and copy the exact text "
                "(including indentation, without line-number prefixes)."
            )
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_text matches {count} places; include more surrounding lines to make "
                "it unique, or set replace_all"
            )
        start = text.index(args.old_text)
        updated = text.replace(args.old_text, args.new_text, -1 if args.replace_all else 1)
        target.write_text(updated, encoding="utf-8")
        first_line = text.count("\n", 0, start) + 1
        snippet = _snippet(updated, first_line, args.new_text.count("\n") + 1)
        where = ctx.workspace.display(target)
        return f'Edited "{where}" ({count if args.replace_all else 1} replacement):\n{snippet}'


class WriteFile(Tool[WriteFileArgs]):
    name = "write_file"
    description = (
        "Create a new file, or replace an existing file entirely, with the given content "
        "(parent directories are created). For changes to part of a file use edit_file."
    )
    side_effects = True
    args_model = WriteFileArgs

    async def run(self, args: WriteFileArgs, ctx: ToolContext) -> str:
        return await asyncio.to_thread(self._run, args, ctx)

    def _run(self, args: WriteFileArgs, ctx: ToolContext) -> str:
        target = ctx.workspace.resolve(args.path)
        if target.is_dir():
            raise ToolError(f'"{args.path}" is a directory')
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        lines = args.content.count("\n") + 1
        return f'{verb} "{ctx.workspace.display(target)}" ({lines} lines)'


def _snippet(text: str, first_line: int, changed_lines: int) -> str:
    """The edited region with a few lines of context, numbered like read_file."""
    lines = text.splitlines()
    start = max(1, first_line - _SNIPPET_CONTEXT_LINES)
    end = min(len(lines), first_line + changed_lines - 1 + _SNIPPET_CONTEXT_LINES)
    return "\n".join(f"{n:>5}\t{lines[n - 1]}" for n in range(start, end + 1))
