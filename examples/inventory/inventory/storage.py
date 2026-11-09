"""JSON file persistence for products and stock movements."""

import json
from datetime import date
from pathlib import Path

from inventory.models import Movement, MovementKind, Product


class JsonRepository:
    """Stores the whole inventory in one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.products: dict[str, Product] = {}
        self.movements: list[Movement] = []

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.products = {p["sku"]: Product(**p) for p in data.get("products", [])}
        self.movements = [
            Movement(
                sku=m["sku"],
                kind=MovementKind(m["kind"]),
                quantity=m["quantity"],
                on=date.fromisoformat(m["on"]),
                note=m.get("note", ""),
            )
            for m in data.get("movements", [])
        ]

    def save(self) -> None:
        data = {
            "products": [vars(p) for p in self.products.values()],
            "movements": [
                {
                    "sku": m.sku,
                    "kind": m.kind.value,
                    "quantity": m.quantity,
                    "on": m.on.isoformat(),
                    "note": m.note,
                }
                for m in self.movements
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    def add_product(self, product: Product) -> None:
        self.products[product.sku] = product

    def get_product(self, sku: str) -> Product:
        try:
            return self.products[sku]
        except KeyError:
            raise KeyError(f"unknown SKU: {sku}") from None

    def add_movement(self, movement: Movement) -> None:
        self.movements.append(movement)
