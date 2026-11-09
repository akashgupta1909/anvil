from datetime import date
from pathlib import Path

import pytest

from inventory.models import Product
from inventory.stock import StockService
from inventory.storage import JsonRepository


@pytest.fixture
def repo(tmp_path: Path) -> JsonRepository:
    repo = JsonRepository(tmp_path / "inventory.json")
    repo.add_product(Product("BOLT", "Bolt", 0.50, reorder_level=10, category="fasteners"))
    repo.add_product(Product("DRILL", "Drill", 89.00, reorder_level=2, category="tools"))
    return repo


@pytest.fixture
def stock(repo: JsonRepository) -> StockService:
    service = StockService(repo)
    service.receive("BOLT", 30, on=date(2026, 1, 5))
    service.receive("DRILL", 5, on=date(2026, 1, 5))
    return service
