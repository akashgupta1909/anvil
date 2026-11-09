from decimal import Decimal

from inventory.reports import inventory_value, stock_report, value_by_category


def test_stock_report_rows(stock):
    rows = stock_report(stock)
    assert [r["sku"] for r in rows] == ["BOLT", "DRILL"]
    assert rows[0]["on_hand"] == 30


def test_inventory_value(stock):
    # 30 x 0.50 + 5 x 89.00
    assert inventory_value(stock) == Decimal("460.00")


def test_value_by_category(stock):
    assert value_by_category(stock) == {
        "fasteners": Decimal("15.00"),
        "tools": Decimal("445.00"),
    }
