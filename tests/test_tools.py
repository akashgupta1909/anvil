import os
from pathlib import Path

import pytest

from anvil.tools.base import ToolError
from anvil.tools.filesystem import (
    ListFiles,
    ListFilesArgs,
    ReadFile,
    ReadFileArgs,
    WriteFile,
    WriteFileArgs,
)
from anvil.tools.python_runner import RunPythonArgs, RunPythonFile
from anvil.tools.workspace import Workspace, WorkspaceManager
from tests.fakes import EXAMPLE, make_ctx


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "pkg/../../x"])
def test_workspace_rejects_escapes(workspace: Workspace, path: str) -> None:
    with pytest.raises(ToolError, match="outside the workspace"):
        workspace.resolve(path)


def test_workspace_rejects_symlink_escape(workspace: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret")
    (workspace.root / "link.txt").symlink_to(secret)
    with pytest.raises(ToolError, match="outside the workspace"):
        workspace.resolve("link.txt")


async def test_list_files(workspace: Workspace) -> None:
    out = await ListFiles().run(ListFilesArgs(directory="pkg"), make_ctx(workspace))
    assert "pkg/calculator.py (" in out
    assert "pkg/render.py (" in out


async def test_read_file_truncates(workspace: Workspace) -> None:
    (workspace.root / "big.txt").write_text("x" * 50)
    out = await ReadFile().run(
        ReadFileArgs(path="big.txt"), make_ctx(workspace, max_output_chars=10)
    )
    assert out.startswith("x" * 10)
    assert "truncated: showing 10 of 50" in out


async def test_read_missing_file(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        await ReadFile().run(ReadFileArgs(path="nope.py"), make_ctx(workspace))


async def test_write_file_creates_parents(workspace: Workspace) -> None:
    out = await WriteFile().run(
        WriteFileArgs(path="new/dir/a.txt", content="hi"), make_ctx(workspace)
    )
    assert (workspace.root / "new/dir/a.txt").read_text() == "hi"
    assert "Wrote 2 characters" in out


async def test_run_python_file(workspace: Workspace) -> None:
    out = await RunPythonFile().run(
        RunPythonArgs(path="main.py", args=["3 + 5"]), make_ctx(workspace)
    )
    assert "exit code: 0" in out
    assert '"result": 8' in out


async def test_run_python_rejects_non_python(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="not a Python"):
        await RunPythonFile().run(RunPythonArgs(path="pkg"), make_ctx(workspace))


async def test_run_python_times_out(workspace: Workspace) -> None:
    (workspace.root / "slow.py").write_text("import time; time.sleep(5)")
    with pytest.raises(ToolError, match="timed out"):
        await RunPythonFile().run(
            RunPythonArgs(path="slow.py"), make_ctx(workspace, run_timeout_seconds=0.5)
        )


async def test_run_python_does_not_leak_secrets(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-should-not-leak")
    (workspace.root / "env.py").write_text("import os; print(dict(os.environ))")
    out = await RunPythonFile().run(RunPythonArgs(path="env.py"), make_ctx(workspace))
    assert "gsk-should-not-leak" not in out
    assert os.environ["GROQ_API_KEY"] == "gsk-should-not-leak"


def test_workspace_manager_copies_and_destroys(tmp_path: Path) -> None:
    manager = WorkspaceManager(EXAMPLE, tmp_path / "root")
    ws = manager.create()
    assert (ws.root / "main.py").is_file()
    manager.destroy(ws)
    assert not ws.root.exists()


def test_schema_is_generated_from_args_model() -> None:
    schema = RunPythonFile().schema()["function"]
    assert schema["name"] == "run_python_file"
    assert schema["parameters"]["required"] == ["path"]
    assert schema["parameters"]["additionalProperties"] is False
    assert "title" not in schema["parameters"]
