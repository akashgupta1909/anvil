"""Non-interactive approval policies (the CLI's interactive prompt lives in cli.py)."""

from __future__ import annotations

from typing import Any


class AllowAll:
    async def approve(self, tool: str, arguments: dict[str, Any]) -> bool:
        return True


class DenyAll:
    """Read-only mode: every side-effecting tool call is refused."""

    async def approve(self, tool: str, arguments: dict[str, Any]) -> bool:
        return False
