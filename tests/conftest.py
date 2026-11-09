"""Shared fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from anvil.config import Settings
from anvil.tools.workspace import Workspace
from tests.fakes import EXAMPLE


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    shutil.copytree(EXAMPLE, root, ignore=shutil.ignore_patterns("__pycache__"))
    return Workspace(root)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "providers": {"p1": {"base_url": "http://p1.test/v1", "api_key_env": "P1_KEY"}},
            "models": {"main": [{"provider": "p1", "model": "m-main"}]},
            "workspace": {"template": EXAMPLE, "root": tmp_path / "workspaces"},
            "tracing": {"dir": tmp_path / "traces"},
            "agent": {"max_steps": 4},
        }
    )
