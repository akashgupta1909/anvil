import pytest

from inventory.importer import CsvImportError, read_products


def test_reads_products(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("sku,name,unit_price,reorder_level\nA1,Widget,2.50,4\n\nB2,Gadget,10,\n")
    products = read_products(path)
    assert [(p.sku, p.unit_price, p.reorder_level) for p in products] == [
        ("A1", 2.5, 4),
        ("B2", 10.0, 0),
    ]


def test_missing_column(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("sku,name\nA1,Widget\n")
    with pytest.raises(CsvImportError, match="unit_price"):
        read_products(path)


def test_bad_price_names_the_line(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("sku,name,unit_price\nA1,Widget,2.50\nB2,Gadget,ten\n")
    with pytest.raises(CsvImportError, match="line 3"):
        read_products(path)
