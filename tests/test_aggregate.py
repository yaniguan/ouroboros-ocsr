"""Aggregation on synthetic run logs (Phase 4 / Am1-D)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ouroboros.eval.aggregate import MixedKindsError, pool_sets, results_table, seed_mean_ci

ROOT = Path(__file__).resolve().parents[1]
ACC = {"A": 0.5, "B": 0.6, "C": 0.7}  # true exact-match rates of the fake arms (rendered set)


def _fake_runs(tmp: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    dirs = []
    for arm in ACC:
        for size in (50_000, 200_000):
            for frac in (0.0, 0.1):
                for seed in range(3):
                    d = tmp / f"{arm}_{size}_{frac}_{seed}"
                    (d / "eval").mkdir(parents=True)
                    meta = {"arm": arm, "size": size, "real_fraction": frac, "seed": seed}
                    (d / "config.yaml").write_text(yaml.safe_dump({"meta": meta, "seed": seed}))
                    with open(d / "eval" / "predictions.jsonl", "w") as f:
                        for set_name, kind, n, base in (
                            ("synthetic_test", "rendered", 400, ACC[arm]),
                            ("uspto_test", "real", 150, 0.2 + frac),
                        ):
                            for angle in range(0, 360, 15):
                                p = base if angle % 90 == 0 else base / 2
                                for i in range(n):
                                    rec = {"set": set_name, "kind": kind, "index": i}
                                    rec.update(angle=float(angle), exact=bool(rng.random() < p))
                                    f.write(json.dumps(rec) + "\n")
                    dirs.append(d)
    return dirs


def _script():
    spec = importlib.util.spec_from_file_location("aggregate", ROOT / "scripts" / "aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_table_plots_and_refusal(tmp_path):
    dirs = _fake_runs(tmp_path / "runs")
    out = tmp_path / "out"
    _script().main(["--runs", *map(str, dirs), "--out", str(out)])
    table = json.loads((out / "results.json").read_text())
    assert table["set_kinds"] == {"synthetic_test": "rendered", "uspto_test": "real"}
    assert len(table["rows"]) == 3 * 2 * 2 and all(r["n_seeds"] == 3 for r in table["rows"])
    for r in table["rows"]:
        c = r["sets"]["synthetic_test"]
        assert c["n_items"] == 400 and c["exact_ci95"][0] <= c["exact_mean"] <= c["exact_ci95"][1]
        assert abs(c["exact_mean"] - ACC[r["arm"]]) < 0.06
        assert r["sets"]["uspto_test"]["n_items"] == 150
    md = (out / "results.md").read_text()
    assert "synthetic_test (rendered)" in md and "uspto_test (real)" in md
    assert "unverified" in md  # parity footnote
    for name in ("size_synthetic_test.png", "real_fraction_uspto_test.png"):
        assert (out / name).stat().st_size > 10_000
    assert len(list(out.glob("rotation_*.png"))) == 2 * 2 * 2
    sweep = json.loads((out / "rotation.json").read_text())
    c = sweep["curves"][0]
    assert len(c["points"]) == 24 and c["mean_on_grid_90"] > c["mean_off_grid_90"]
    # averaging rendered with real is refused; same-kind averaging is allowed
    with pytest.raises(MixedKindsError):
        _script().main(
            [
                "--runs",
                *map(str, dirs),
                "--out",
                str(out),
                "--average",
                "synthetic_test",
                "uspto_test",
            ]
        )
    with pytest.raises(MixedKindsError):
        pool_sets(table["set_kinds"], ["synthetic_test", "uspto_test"])
    assert pool_sets(table["set_kinds"], ["synthetic_test"]) == ["synthetic_test"]


def test_ci_requirements():
    with pytest.raises(ValueError):
        results_table([], n_boot=500)
    m = np.array([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=float)
    mean, lo, hi = seed_mean_ci(m, 1000)
    assert mean == 0.75 and lo <= mean <= hi
