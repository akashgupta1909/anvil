"""Content hashes: identify exact prompt versions in traces and detect workspace changes."""

from __future__ import annotations

import hashlib
from pathlib import Path

_IGNORED_DIRS = {"__pycache__"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint_dir(root: Path) -> dict[str, str]:
    """Map each file's relative path to its content hash (caches ignored)."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _IGNORED_DIRS.intersection(path.relative_to(root).parts)
    }
