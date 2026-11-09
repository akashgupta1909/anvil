"""Domain types for the inventory system."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class MovementKind(Enum):
    RECEIVE = "receive"
    SHIP = "ship"
    ADJUST = "adjust"


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    unit_price: float
    reorder_level: int = 0
    category: str = "general"


@dataclass(frozen=True)
class Movement:
    sku: str
    kind: MovementKind
    quantity: int
    on: date = field(default_factory=date.today)
    note: str = ""

    def signed_quantity(self) -> int:
        """Positive for stock coming in, negative for stock going out."""
        if self.kind is MovementKind.SHIP:
            return -self.quantity
        return self.quantity


class InventoryError(Exception):
    """Raised for invalid operations, e.g. shipping more than is in stock."""
