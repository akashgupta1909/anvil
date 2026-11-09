import os
import sys
from pathlib import Path

import pytest

from anvil.tools.base import ToolError
from anvil.tools.changes import ShowChanges, ShowChangesArgs, changed_files
from anvil.tools.commands import RunCommand, RunCommandArgs
from anvil.tools.filesystem import (
    EditFile,
    EditFileArgs,
    ListFiles,
    ListFilesArgs,
    ReadFile,
    ReadFileArgs,
    WriteFile,
    WriteFileArgs,
)
from anvil.tools.search import FindFiles, FindFilesArgs, SearchCode, SearchCodeArgs
from anvil.tools.workspace import Workspace, WorkspaceManager
from tests.fakes import EXAMPLE, make_ctx

COMMANDS = {"python": ["python"], "pytest": ["python", "-m", "pytest", "-p", "no:cacheprovider"]}


# --- workspace -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "inventory/../../x"])
def test_workspace_rejects_escapes(workspace: Workspace, path: str) -> None:
    with pytest.raises(ToolError, match="outside the workspace"):
        workspace.resolve(path)


def test_workspace_rejects_symlink_escape(workspace: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret")
    (workspace.root / "link.txt").symlink_to(secret)
    with pytest.raises(ToolError, match="outside the workspace"):
        workspace.resolve("link.txt")
    assert "link.txt" not in workspace.snapshot()  # nor reachable by iteration


def test_snapshot_skips_caches_and_binaries(workspace: Workspace) -> None:
    (workspace.root / ".pytest_cache").mkdir()
    (workspace.root / ".pytest_cache" / "x").write_text("cache")
    (workspace.root / "blob.bin").write_bytes(b"\0\1\2")
    snapshot = workspace.snapshot()
    assert "inventory/stock.py" in snapshot
    assert not [p for p in snapshot if p.startswith(".pytest_cache") or p == "blob.bin"]


def test_workspace_manager_copies_and_destroys(tmp_path: Path) -> None:
    manager = WorkspaceManager(EXAMPLE, tmp_path / "root")
    ws = manager.create()
    assert (ws.root / "inventory" / "cli.py").is_file()
    manager.destroy(ws)
    assert not ws.root.exists()


# --- list / find / search ------------------------------------------------------------------


async def test_list_files(workspace: Workspace) -> None:
    out = await ListFiles().run(ListFilesArgs(directory="inventory"), make_ctx(workspace))
    assert "inventory/stock.py (" in out
    assert "__pycache__" not in out


async def test_find_files_with_globstar(workspace: Workspace) -> None:
    out = await FindFiles().run(FindFilesArgs(pattern="**/test_*.py"), make_ctx(workspace))
    assert "tests/test_stock.py" in out.splitlines()
    out = await FindFiles().run(FindFilesArgs(pattern="**/*.toml"), make_ctx(workspace))
    assert out == "pyproject.toml"  # '**/' also matches the root


async def test_search_code(workspace: Workspace) -> None:
    out = await SearchCode().run(
        SearchCodeArgs(pattern=r"def needs_\w+", file_glob="*.py"), make_ctx(workspace)
    )
    assert out.startswith("inventory/stock.py:")
    assert "def needs_reorder" in out


async def test_search_code_scoped_and_case_insensitive(workspace: Workspace) -> None:
    out = await SearchCode().run(
        SearchCodeArgs(pattern="CSVIMPORTERROR", path="tests", ignore_case=True),
        make_ctx(workspace),
    )
    assert out.splitlines()
    assert all(line.startswith("tests/") for line in out.splitlines())


async def test_search_code_rejects_bad_regex(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="invalid regular expression"):
        await SearchCode().run(SearchCodeArgs(pattern="("), make_ctx(workspace))


# --- read / edit / write -------------------------------------------------------------------


async def test_read_file_numbers_lines_and_supports_ranges(workspace: Workspace) -> None:
    out = await ReadFile().run(
        ReadFileArgs(path="inventory/stock.py", start_line=2, end_line=3), make_ctx(workspace)
    )
    assert out.splitlines() == ["    2\t", "    3\tfrom datetime import date"]


async def test_read_file_cuts_off_long_files_with_a_hint(workspace: Workspace) -> None:
    (workspace.root / "big.txt").write_text("\n".join(f"line {i}" for i in range(1, 200)))
    out = await ReadFile().run(
        ReadFileArgs(path="big.txt"), make_ctx(workspace, max_output_chars=100)
    )
    assert out.splitlines()[0] == "    1\tline 1"
    assert "continue with start_line=" in out


async def test_read_missing_file(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        await ReadFile().run(ReadFileArgs(path="nope.py"), make_ctx(workspace))


async def test_edit_file_replaces_exactly_one_match(workspace: Workspace) -> None:
    args = EditFileArgs(
        path="inventory/pricing.py",
        old_text='TAX_RATE = Decimal("0.08")',
        new_text='TAX_RATE = Decimal("0.2")',
    )
    out = await EditFile().run(args, make_ctx(workspace))
    assert 'TAX_RATE = Decimal("0.2")' in (workspace.root / "inventory/pricing.py").read_text()
    assert "1 replacement" in out
    assert 'TAX_RATE = Decimal("0.2")' in out  # snippet of the edited region


async def test_edit_file_rejects_missing_and_ambiguous_text(workspace: Workspace) -> None:
    ctx = make_ctx(workspace)
    with pytest.raises(ToolError, match="not found"):
        await EditFile().run(
            EditFileArgs(path="inventory/stock.py", old_text="nope", new_text="x"), ctx
        )
    with pytest.raises(ToolError, match="matches 2 places"):
        await EditFile().run(
            EditFileArgs(
                path="inventory/stock.py", old_text="self.repo.get_product(sku)", new_text="x"
            ),
            ctx,
        )


async def test_edit_file_replace_all(workspace: Workspace) -> None:
    args = EditFileArgs(
        path="inventory/stock.py",
        old_text="self.repo.get_product(sku)",
        new_text="self.repo.get_product(sku)  # check",
        replace_all=True,
    )
    out = await EditFile().run(args, make_ctx(workspace))
    assert "2 replacement" in out


async def test_write_file_creates_parents(workspace: Workspace) -> None:
    out = await WriteFile().run(
        WriteFileArgs(path="new/dir/a.txt", content="hi"), make_ctx(workspace)
    )
    assert (workspace.root / "new/dir/a.txt").read_text() == "hi"
    assert out.startswith('Created "new/dir/a.txt"')


# --- show_changes --------------------------------------------------------------------------


async def test_show_changes_reports_edits_new_and_deleted_files(workspace: Workspace) -> None:
    ctx = make_ctx(workspace, baseline=workspace.snapshot())
    assert await ShowChanges().run(ShowChangesArgs(), ctx) == "No changes since the task started."
    (workspace.root / "inventory/models.py").write_text("changed\n")
    (workspace.root / "new.txt").write_text("hello\n")
    (workspace.root / "README.md").unlink()
    out = await ShowChanges().run(ShowChangesArgs(), ctx)
    assert "+++ b/inventory/models.py" in out
    assert "+hello" in out
    assert "+++ /dev/null" in out  # README deleted
    scoped = await ShowChanges().run(ShowChangesArgs(path="inventory"), ctx)
    assert "new.txt" not in scoped
    assert changed_files(workspace, ctx.baseline) == [
        "README.md",
        "inventory/models.py",
        "new.txt",
    ]


# --- run_command ---------------------------------------------------------------------------


async def test_run_command_runs_project_cli(workspace: Workspace) -> None:
    out = await RunCommand(COMMANDS, 30).run(
        RunCommandArgs(command=["python", "-m", "inventory", "value"]), make_ctx(workspace)
    )
    assert "exit code: 0" in out
    assert "total value: 643.25" in out


async def test_run_command_runs_tests_without_cache(workspace: Workspace) -> None:
    out = await RunCommand(COMMANDS, 60).run(
        RunCommandArgs(command=["pytest"]), make_ctx(workspace)
    )
    assert "20 passed" in out
    assert not (workspace.root / ".pytest_cache").exists()


async def test_run_command_rejects_programs_not_allowlisted(workspace: Workspace) -> None:
    tool = RunCommand(COMMANDS, 30)
    assert "Allowed programs: pytest, python" in tool.description
    with pytest.raises(ToolError, match="'curl' is not an allowed program"):
        await tool.run(RunCommandArgs(command=["curl", "x"]), make_ctx(workspace))


async def test_run_command_accepts_a_string(workspace: Workspace) -> None:
    out = await RunCommand(COMMANDS, 30).run(
        RunCommandArgs(command="python -m inventory value"), make_ctx(workspace)
    )
    assert "total value: 643.25" in out


@pytest.mark.parametrize(
    ("command", "error"),
    [
        ("cd tests && pytest", "shell syntax"),
        (["cd", "tests"], "no cd needed"),
        ("pytest | tail", "shell syntax"),
        ("", "empty"),
    ],
)
async def test_run_command_explains_shell_misuse(
    workspace: Workspace, command: str | list[str], error: str
) -> None:
    with pytest.raises(ToolError, match=error):
        await RunCommand(COMMANDS, 30).run(RunCommandArgs(command=command), make_ctx(workspace))


async def test_run_command_times_out(workspace: Workspace) -> None:
    (workspace.root / "slow.py").write_text("import time; time.sleep(5)")
    with pytest.raises(ToolError, match="timed out"):
        await RunCommand(COMMANDS, 0.5).run(
            RunCommandArgs(command=["python", "slow.py"]), make_ctx(workspace)
        )


async def test_run_command_does_not_leak_secrets(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-should-not-leak")
    out = await RunCommand(COMMANDS, 30).run(
        RunCommandArgs(command=["python", "-c", "import os; print(dict(os.environ))"]),
        make_ctx(workspace),
    )
    assert "gsk-should-not-leak" not in out
    assert os.environ["GROQ_API_KEY"] == "gsk-should-not-leak"


async def test_run_command_keeps_head_and_tail_of_long_output(workspace: Workspace) -> None:
    code = "print('START'); print('x' * 5000); print('END')"
    out = await RunCommand(COMMANDS, 30).run(
        RunCommandArgs(command=["python", "-c", code]), make_ctx(workspace, max_output_chars=200)
    )
    assert "START" in out
    assert "END" in out
    assert "characters omitted" in out


async def test_python_in_allowlist_means_this_interpreter(workspace: Workspace) -> None:
    tool = RunCommand({"python": ["python", "-c", "import sys; print(sys.executable)"]}, 30)
    out = await tool.run(RunCommandArgs(command=["python"]), make_ctx(workspace))
    assert sys.executable in out


def test_schema_is_generated_from_args_model() -> None:
    schema = EditFile().schema()["function"]
    assert schema["name"] == "edit_file"
    assert schema["parameters"]["required"] == ["path", "old_text", "new_text"]
    assert schema["parameters"]["additionalProperties"] is False
    assert "title" not in schema["parameters"]
