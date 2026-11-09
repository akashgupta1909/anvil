import pytest

from inventory.models import InventoryError


def test_level_sums_movements(stock):
    stock.ship("BOLT", 12)
    assert stock.level("BOLT") == 18


def test_cannot_ship_more_than_on_hand(stock):
    with pytest.raises(InventoryError, match="only 30 in stock"):
        stock.ship("BOLT", 31)


def test_quantity_must_be_positive(stock):
    with pytest.raises(InventoryError):
        stock.receive("BOLT", 0)


def test_unknown_sku(stock):
    with pytest.raises(KeyError, match="NOPE"):
        stock.level("NOPE")


def test_reorder_when_below_level(stock, repo):
    stock.ship("DRILL", 4)  # 1 left, reorder level 2
    assert [p.sku for p in stock.products_to_reorder()] == ["DRILL"]


def test_no_reorder_when_well_stocked(stock):
    assert stock.products_to_reorder() == []
