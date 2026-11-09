"""Import every module in its own fresh interpreter to catch circular imports that a
warm test process (where modules are already cached) would hide."""

import pkgutil
import subprocess
import sys

import pytest

import anvil

MODULES = sorted(m.name for m in pkgutil.walk_packages(anvil.__path__, prefix="anvil."))


def test_discovers_all_modules() -> None:
    assert "anvil.config" in MODULES


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_fresh_interpreter(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
