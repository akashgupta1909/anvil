"""The sandbox root that every tool path is resolved against."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from anvil.tools.base import ToolError


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        """Resolve ``relative`` inside the workspace, following symlinks, so neither
        ``../``, absolute paths nor a symlink pointing outside can escape it."""
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(f'"{relative}" is outside the workspace')
        return target

    def display(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


class WorkspaceManager:
    """Creates disposable copies of a template project (per API session / eval task)."""

    def __init__(self, template: Path, root: Path) -> None:
        self._template = template
        self._root = root

    def create(self) -> Workspace:
        self._root.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="ws-", dir=self._root))
        shutil.copytree(
            self._template,
            path,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        return Workspace(path)

    def destroy(self, workspace: Workspace) -> None:
        shutil.rmtree(workspace.root, ignore_errors=True)
