from typing import Any

import pytest

from anvil.models import ToolCall
from anvil.tools.approval import DenyAll
from anvil.tools.filesystem import ListFiles, ReadFile, WriteFile
from anvil.tools.registry import ToolRegistry
from anvil.tools.workspace import Workspace
from tests.fakes import make_ctx


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([ListFiles(), ReadFile(), WriteFile()])


def call(name: str, arguments: str) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


async def test_success(registry: ToolRegistry, workspace: Workspace) -> None:
    out = await registry.execute(call("read_file", '{"path": "README.md"}'), make_ctx(workspace))
    assert out.record.ok
    assert out.record.arguments == {"path": "README.md"}
    assert "inventory" in out.content


@pytest.mark.parametrize(
    ("name", "arguments", "error_type"),
    [
        ("delete_everything", "{}", "unknown_tool"),
        ("read_file", '{"path": ', "bad_arguments"),
        ("read_file", "{}", "bad_arguments"),
        ("read_file", '{"path": "README.md", "extra": 1}', "bad_arguments"),
        ("read_file", '{"path": "../x"}', "tool_error"),
    ],
)
async def test_model_errors_become_tool_results(
    registry: ToolRegistry, workspace: Workspace, name: str, arguments: str, error_type: str
) -> None:
    out = await registry.execute(call(name, arguments), make_ctx(workspace))
    assert not out.record.ok
    assert out.record.error_type == error_type
    assert out.content.startswith("Error:")


async def test_side_effects_need_approval(registry: ToolRegistry, workspace: Workspace) -> None:
    out = await registry.execute(
        call("write_file", '{"path": "a.txt", "content": "x"}'), make_ctx(workspace, DenyAll())
    )
    assert out.record.error_type == "denied"
    assert not (workspace.root / "a.txt").exists()


async def test_read_only_tools_skip_approval(registry: ToolRegistry, workspace: Workspace) -> None:
    out = await registry.execute(call("list_files", "{}"), make_ctx(workspace, DenyAll()))
    assert out.record.ok


async def test_approver_receives_arguments(registry: ToolRegistry, workspace: Workspace) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    class Recorder:
        async def approve(self, tool: str, arguments: dict[str, Any]) -> bool:
            seen.append((tool, arguments))
            return True

    await registry.execute(
        call("write_file", '{"path": "a.txt", "content": "x"}'), make_ctx(workspace, Recorder())
    )
    assert seen == [("write_file", {"path": "a.txt", "content": "x"})]


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([ReadFile(), ReadFile()])
