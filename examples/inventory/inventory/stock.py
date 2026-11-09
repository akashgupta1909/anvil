"""Stock levels and stock movements."""

from datetime import date

from inventory.models import InventoryError, Movement, MovementKind, Product
from inventory.storage import JsonRepository


class StockService:
    def __init__(self, repo: JsonRepository) -> None:
        self.repo = repo

    def level(self, sku: str) -> int:
        """Current quantity on hand for ``sku``."""
        self.repo.get_product(sku)
        return sum(m.signed_quantity() for m in self.repo.movements if m.sku == sku)

    def receive(self, sku: str, quantity: int, on: date | None = None, note: str = "") -> None:
        self._record(sku, MovementKind.RECEIVE, quantity, on, note)

    def ship(self, sku: str, quantity: int, on: date | None = None, note: str = "") -> None:
        available = self.level(sku)
        if quantity > available:
            raise InventoryError(f"cannot ship {quantity} x {sku}: only {available} in stock")
        self._record(sku, MovementKind.SHIP, quantity, on, note)

    def needs_reorder(self, product: Product) -> bool:
        """A product needs reordering once stock falls to its reorder level or below."""
        return self.level(product.sku) <= product.reorder_level

    def products_to_reorder(self) -> list[Product]:
        return [p for p in self.repo.products.values() if self.needs_reorder(p)]

    def _record(
        self, sku: str, kind: MovementKind, quantity: int, on: date | None, note: str
    ) -> None:
        if quantity <= 0:
            raise InventoryError("quantity must be positive")
        self.repo.get_product(sku)
        self.repo.add_movement(Movement(sku, kind, quantity, on or date.today(), note))
