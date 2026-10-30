"""scripts/image_to_forcefield.py on a predictions CSV, with MACE unavailable."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ase_io = pytest.importorskip("ase.io")

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "image_to_forcefield", ROOT / "scripts" / "image_to_forcefield.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "subset"])
        w.writeheader()
        for rid, smi in rows:
            w.writerow({"id": rid, "gold": smi, "pred": smi, "subset": "UOB"})
    return path


@pytest.fixture
def no_mace(monkeypatch):
    mod = _script()

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'mace'")

    monkeypatch.setattr(mod, "mace_calculator", missing)
    return mod


def test_csv_without_mace_writes_geometry_and_null_energy(tmp_path, no_mace):
    preds = _csv(tmp_path / "preds.csv", [("1", "CCO"), ("2", "CC(N)C(=O)O")])
    out = tmp_path / "ff" / "ocsr.extxyz"
    summary = no_mace.main(["--predictions", str(preds), "--out", str(out), "--n-conf", "2"])
    assert summary["mace_model"] is None and summary["rows"] == {"ok": 2}

    frames = ase_io.read(out, index=":", format="extxyz")
    assert len(frames) == summary["frames_written"] >= 2
    # extxyz reads numeric-looking strings back as numbers: compare ids as strings
    assert {str(f.info["source_id"]) for f in frames} == {"1", "2"}
    for f in frames:
        assert f.positions.shape == (len(f), 3) and np.isfinite(f.positions).all()
        assert np.abs(f.positions).max() > 0.1  # real 3D coordinates
        assert not f.info["has_energy"]
        assert "mace_energy" not in f.info and "mace_forces" not in f.arrays
        assert f.calc is None  # no energy/forces smuggled in as a calculator
        assert f.info["source_image"] == f"UOB/{f.info['source_id']}"
        assert len(f.info["inchikey"]) == 27
    ethanol = [f for f in frames if str(f.info["source_id"]) == "1"][0]
    assert sorted(ethanol.get_chemical_symbols()) == sorted("CCOHHHHHH")

    records = [json.loads(x) for x in out.with_name("ocsr.manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == ["1", "2"]
    assert all(r["energy"] is None and r["status"] == "ok" for r in records)
    assert all(c["energy"] is None for r in records for c in r["conformers"])
    by_id = {r["id"]: r for r in records}
    assert by_id["1"]["stereo_unassigned"] is False
    # alanine drawn without a wedge: the predicted SMILES leaves its stereocentre unassigned
    assert by_id["2"]["stereo_unassigned"] is True
    assert by_id["2"]["n_unassigned_stereocentres"] == 1


def test_invalid_rgroup_and_duplicates_are_recorded_not_written(tmp_path, no_mace):
    out = tmp_path / "ocsr.extxyz"
    first = _csv(tmp_path / "a.csv", [("1", "CCO"), ("2", "C1CC"), ("3", "*c1ccccc1"),
                                      ("4", "OCC")])
    no_mace.main(["--predictions", str(first), "--out", str(out), "--n-conf", "1",
                  "--dedupe-inchikey", "--only-valid"])
    # a second run appends, and dedupes against the existing manifest
    second = _csv(tmp_path / "b.csv", [("5", "C(C)O"), ("6", "c1ccccc1")])
    no_mace.main(["--predictions", str(second), "--out", str(out), "--n-conf", "1",
                  "--dedupe-inchikey"])
    records = [json.loads(x) for x in out.with_name("ocsr.manifest.jsonl").read_text().splitlines()]
    status = {r["id"]: r["status"] for r in records}
    assert status["1"] == "ok" and status["6"] == "ok"
    assert status["2"].startswith("invalid_smiles")
    assert status["3"] == "pseudo_atoms"
    assert status["4"] == "duplicate_inchikey" and status["5"] == "duplicate_inchikey"
    frames = ase_io.read(out, index=":", format="extxyz")
    assert sorted(str(f.info["source_id"]) for f in frames) == ["1", "6"]
