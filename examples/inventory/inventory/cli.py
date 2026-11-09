"""Command-line interface: python -m inventory <command> ..."""

import argparse
import sys
from pathlib import Path

from inventory.importer import CsvImportError, read_products
from inventory.models import InventoryError
from inventory.reports import inventory_value, stock_report
from inventory.stock import StockService
from inventory.storage import JsonRepository

DEFAULT_DB = Path("data/inventory.json")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = JsonRepository(args.db)
    repo.load()
    stock = StockService(repo)
    try:
        if args.command == "import":
            for product in read_products(args.csv):
                repo.add_product(product)
            repo.save()
            print(f"imported products from {args.csv}")
        elif args.command == "receive":
            stock.receive(args.sku, args.quantity)
            repo.save()
            print(f"{args.sku}: {stock.level(args.sku)} on hand")
        elif args.command == "ship":
            stock.ship(args.sku, args.quantity)
            repo.save()
            print(f"{args.sku}: {stock.level(args.sku)} on hand")
        elif args.command == "report":
            for row in stock_report(stock):
                flag = "  REORDER" if row["reorder"] else ""
                print(f"{row['sku']:<8} {row['name']:<24} {row['on_hand']:>5}{flag}")
        elif args.command == "value":
            print(f"total value: {inventory_value(stock)}")
    except (InventoryError, CsvImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inventory")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="inventory JSON file")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="import products from a CSV file")
    imp.add_argument("csv", type=Path)
    for name in ("receive", "ship"):
        move = sub.add_parser(name, help=f"{name} stock")
        move.add_argument("sku")
        move.add_argument("quantity", type=int)
    sub.add_parser("report", help="stock levels and reorder flags")
    sub.add_parser("value", help="total value of stock on hand")
    return parser
