"""Read-only reports over the current inventory."""

from decimal import Decimal

from inventory.pricing import CENT
from inventory.stock import StockService


def stock_report(stock: StockService) -> list[dict[str, object]]:
    """One row per product, sorted by SKU."""
    rows: list[dict[str, object]] = []
    for sku in sorted(stock.repo.products):
        product = stock.repo.products[sku]
        rows.append(
            {
                "sku": sku,
                "name": product.name,
                "on_hand": stock.level(sku),
                "reorder": stock.needs_reorder(product),
            }
        )
    return rows


def inventory_value(stock: StockService) -> Decimal:
    """Value of all stock on hand at unit price."""
    total = Decimal("0")
    for product in stock.repo.products.values():
        total += Decimal(str(product.unit_price)) * stock.level(product.sku)
    return total.quantize(CENT)


def value_by_category(stock: StockService) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {}
    for product in stock.repo.products.values():
        value = Decimal(str(product.unit_price)) * stock.level(product.sku)
        totals[product.category] = totals.get(product.category, Decimal("0")) + value
    return {category: value.quantize(CENT) for category, value in sorted(totals.items())}
