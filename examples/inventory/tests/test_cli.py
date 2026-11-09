import shutil
from pathlib import Path

from inventory.cli import main

DATA = Path(__file__).resolve().parents[1] / "data"


def test_report_flags_reorder(tmp_path, capsys):
    db = tmp_path / "inv.json"
    shutil.copy(DATA / "inventory.json", db)
    assert main(["--db", str(db), "report"]) == 0
    out = capsys.readouterr().out
    assert "BOLT-M6" in out
    assert "DRILL-18V" in out and "REORDER" in out


def test_ship_too_many_is_an_error(tmp_path, capsys):
    db = tmp_path / "inv.json"
    shutil.copy(DATA / "inventory.json", db)
    assert main(["--db", str(db), "ship", "GLOVE-L", "999"]) == 1
    assert "only 28 in stock" in capsys.readouterr().err
