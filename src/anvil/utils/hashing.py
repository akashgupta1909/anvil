"""Content hashes: identify the exact prompt text used in a trace."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
