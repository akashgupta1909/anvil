"""The sandbox root that every tool path is resolved against."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from anvil.tools.base import ToolError

# Never listed, searched, snapshotted or diffed: VCS data, environments and caches.
IGNORED_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
)
_MAX_SNAPSHOT_FILE_BYTES = 256_000


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

    def iter_files(self, start: Path | None = None) -> Iterator[Path]:
        """Files under ``start`` (default: the root) in sorted order, skipping ignored
        directories and anything that resolves outside the workspace."""
        base = start or self.root
        if base.is_file():
            yield base
            return
        for path in sorted(base.rglob("*")):
            rel_parts = path.relative_to(self.root).parts
            if IGNORED_DIRS.intersection(rel_parts) or not path.is_file():
                continue
            if path.resolve().is_relative_to(self.root):
                yield path

    def snapshot(self) -> dict[str, str]:
        """Text content of every (reasonably small, non-binary) file, keyed by path.
        Taken at the start of a run so changes can be diffed and reported."""
        files: dict[str, str] = {}
        for path in self.iter_files():
            text = read_text_file(path, _MAX_SNAPSHOT_FILE_BYTES)
            if text is not None:
                files[self.display(path)] = text
        return files


def read_text_file(path: Path, max_bytes: int) -> str | None:
    """File content as text, or None if it is too large or looks binary."""
    try:
        if path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


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
            ignore=shutil.ignore_patterns(*IGNORED_DIRS),
        )
        return Workspace(path)

    def destroy(self, workspace: Workspace) -> None:
        shutil.rmtree(workspace.root, ignore_errors=True)
