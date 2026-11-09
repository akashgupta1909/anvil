"""Import products from CSV files."""

import csv
from pathlib import Path

from inventory.models import Product


class CsvImportError(ValueError):
    """A row in the CSV file is invalid."""


REQUIRED_COLUMNS = {"sku", "name", "unit_price"}


def read_products(path: Path) -> list[Product]:
    """Read products from a CSV file with a header row.

    Required columns: sku, name, unit_price. Optional: reorder_level, category.
    Blank lines are skipped. Raises CsvImportError naming the line of a bad row.
    """
    products: list[Product] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise CsvImportError(f"missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            products.append(_parse_row(row, reader.line_num))
    return products


def _parse_row(row: dict[str, str], line: int) -> Product:
    try:
        return Product(
            sku=row["sku"].strip(),
            name=row["name"].strip(),
            unit_price=float(row["unit_price"]),
            reorder_level=int(row.get("reorder_level") or 0),
            category=(row.get("category") or "general").strip(),
        )
    except ValueError as exc:
        raise CsvImportError(f"line {line}: {exc}") from exc
