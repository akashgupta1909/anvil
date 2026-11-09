# inventory

A small inventory management package: products, stock movements, bulk pricing with tax,
CSV import and reports, with a command-line interface.

```bash
python -m inventory report          # stock levels, flags products to reorder
python -m inventory value           # value of stock on hand
python -m inventory receive BOLT-M6 10
python -m inventory ship BOLT-M6 5
python -m inventory import data/products.csv
python -m pytest                    # run the tests
```

Data lives in `data/inventory.json` (override with `--db`).
