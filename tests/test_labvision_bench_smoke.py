import json

from scripts.labvision_bench import HEADER, main


def test_bench_smoke(tmp_path):
    res = main(["--out", str(tmp_path), "--n", "4", "--sweep-n", "1", "--calib-n", "2"])
    md = (tmp_path / "labvision_bench.md").read_text()
    data = json.loads((tmp_path / "labvision_bench.json").read_text())
    assert data["header"] == HEADER
    n_tables = sum(line.startswith("|---") for line in md.splitlines())
    assert n_tables >= 5 and md.count(HEADER) >= n_tables + 1
    for kind in ("rect", "circ"):
        assert res[kind]["sources"]["pixel"]["n_images"] == 4
        assert 0.0 <= res[kind]["sources"]["pixel"]["well_id_acc"] <= 1.0
