from decimal import Decimal

from inventory.pricing import bulk_discount_rate, line_total, order_total


def test_no_discount_below_first_tier():
    assert bulk_discount_rate(49) == Decimal("0")


def test_tiers():
    assert bulk_discount_rate(50) == Decimal("0.05")
    assert bulk_discount_rate(100) == Decimal("0.10")


def test_line_total_applies_discount():
    assert line_total(2.00, 100) == Decimal("180.00")


def test_order_total_adds_tax():
    # 10 x 9.99 = 99.90, tax 8% = 7.99 (7.992 rounded)
    assert order_total([(9.99, 10)]) == Decimal("107.89")
