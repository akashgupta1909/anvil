from inventory.storage import JsonRepository


def test_round_trip(stock, repo):
    stock.ship("BOLT", 3)
    repo.save()
    loaded = JsonRepository(repo.path)
    loaded.load()
    assert loaded.products == repo.products
    assert loaded.movements == repo.movements


def test_missing_file_loads_empty(tmp_path):
    repo = JsonRepository(tmp_path / "none.json")
    repo.load()
    assert repo.products == {}
