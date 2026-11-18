"""Tests for the untouched frozen-test-v2 manifest, leakage gate, and subset fetcher."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image
from rdkit import Chem

from ouroboros.data.real import (
    EXPECTED_DEV_TEST_MANIFEST_SHA256,
    EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256,
    FROZEN_TEST_V2_COUNTS,
    FROZEN_TEST_V2_MANIFEST,
    FROZEN_TEST_V2_SEED,
    ingest,
    read_real_manifest,
)
from ouroboros.eval.standardize import leakage_key, standardize
from ouroboros.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _png_bytes(shade: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (shade, shade, shade)).save(buf, format="PNG")
    return buf.getvalue()


def test_frozen_test_v2_manifest_hash_and_counts():
    manifest = ROOT / FROZEN_TEST_V2_MANIFEST
    assert manifest.is_file()
    raw = manifest.read_bytes()
    assert b"\r\n" not in raw, "Committed frozen manifest must use LF line endings"
    assert file_sha256(manifest) == EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256
    assert EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256 != EXPECTED_DEV_TEST_MANIFEST_SHA256
    assert FROZEN_TEST_V2_SEED == 2026

    fetch_mod = _load_script("fetch_real_subset")
    assert fetch_mod.EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256 == (
        EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256
    )

    rows = read_real_manifest(manifest)
    assert len(rows) == 300
    assert dict(Counter(r["source"] for r in rows)) == FROZEN_TEST_V2_COUNTS
    assert all(r["split"] == "test" for r in rows)

    keys = [leakage_key(standardize(r["smiles"])) for r in rows]
    assert all(k is not None for k in keys)
    assert len(set(keys)) == 300
    assert len({k[:14] for k in keys if k is not None}) == 300

    folder = manifest.parent
    image_files = [
        p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}
    ]
    assert image_files == []

    receipt = json.loads((folder / "leakage.json").read_text())
    assert {r["train"] for r in receipt} == {"subset100", "pmc_train", "pmc_val", "zinc250k"}
    assert all(
        r["n_eval"] == 300 and r["overlap_inchikey"] == 0 and r["overlap_connectivity"] == 0
        for r in receipt
    )


def test_leakage_gate_on_frozen_manifest_csv(tmp_path):
    leak = _load_script("check_leakage")
    manifest = ROOT / FROZEN_TEST_V2_MANIFEST
    rows = read_real_manifest(manifest)

    # Disjoint synthetic molecule not in frozen-test-v2 (simple short chain: methane)
    clean_csv = tmp_path / "clean.csv"
    clean_csv.write_text("smiles,split\nC,train\nCC,val\n")
    frozen_k14 = {
        leakage_key(standardize(r["smiles"]))[:14]  # type: ignore[index]
        for r in rows
    }
    assert leakage_key(standardize("C"))[:14] not in frozen_k14  # type: ignore[index]
    assert leakage_key(standardize("CC"))[:14] not in frozen_k14  # type: ignore[index]

    rc = leak.main(
        [
            "--eval",
            f"frozen_v2={manifest}",
            "--train",
            f"tr={clean_csv}:train",
            f"va={clean_csv}:val",
            "--fail-on-connectivity",
        ]
    )
    assert rc == 0

    # Pick a chiral molecule from frozen-test-v2, strip stereo -> connectivity block still matches
    chiral_smi = next(r["smiles"] for r in rows if "@" in standardize(r["smiles"]).smiles)
    mol = Chem.MolFromSmiles(standardize(chiral_smi).smiles)
    Chem.RemoveStereochemistry(mol)
    flat_smi = Chem.MolToSmiles(mol)

    leaky_csv = tmp_path / "leaky.csv"
    leaky_csv.write_text(f"smiles,split\n{flat_smi},train\n")
    assert (
        leak.main(
            [
                "--eval",
                f"frozen_v2={manifest}",
                "--train",
                f"leaky={leaky_csv}:train",
                "--fail-on-connectivity",
            ]
        )
        == 1
    )


def test_fetch_real_subset_exclusion_and_from_manifest(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    fetch_mod = _load_script("fetch_real_subset")

    cache = tmp_path / "cache"
    cache.mkdir()
    records = [
        ("id0", "CCO"),  # excluded via old manifest
        ("id1", "C[C@@H](N)C(=O)O"),  # enantiomer of excluded key
        ("id2", "not_a_smiles("),  # invalid gold SMILES
        ("id3", "c1ccccc1O"),  # clean 1
        ("id4", "CC(=O)Oc1ccccc1C(=O)O"),  # clean 2
    ]
    table = pa.table(
        {
            "image": [{"bytes": _png_bytes(200 + i), "path": None} for i in range(len(records))],
            "image_id": [r[0] for r in records],
            "SMILES": [r[1] for r in records],
        }
    )
    pq.write_table(table, cache / "MINI.parquet")

    old_manifest = tmp_path / "old.csv"
    old_manifest.write_text("image,smiles,source,split,id\nimages/MINI/id0.png,CCO,MINI,test,id0\n")
    excl_keys = tmp_path / "excl.txt"
    ala_key = leakage_key(standardize("C[C@H](N)C(=O)O"))
    assert ala_key is not None
    excl_keys.write_text(ala_key + "\n")

    frozen_dir = tmp_path / "frozen_out"
    fetch_mod.main(
        [
            "--sets",
            "MINI",
            "--n",
            "2",
            "--seed",
            "7",
            "--cache-dir",
            str(cache),
            "--exclude-manifest",
            str(old_manifest),
            "--exclude-keys",
            str(excl_keys),
            "--lf",
            "--manifest-only",
            "--out",
            str(frozen_dir),
        ]
    )
    frozen_csv = frozen_dir / "manifest.csv"
    assert frozen_csv.is_file()
    assert not (frozen_dir / "images").exists()
    assert b"\r\n" not in frozen_csv.read_bytes()

    with open(frozen_csv, newline="") as f:
        picked = list(csv.DictReader(f))
    assert [r["id"] for r in picked] == ["id3", "id4"]

    sha = file_sha256(frozen_csv)
    mat_dir = tmp_path / "materialized"
    fetch_mod.main(
        [
            "--from-manifest",
            str(frozen_csv),
            "--expected-sha256",
            sha,
            "--cache-dir",
            str(cache),
            "--out",
            str(mat_dir),
        ]
    )
    assert file_sha256(mat_dir / "manifest.csv") == sha
    stats = ingest(mat_dir / "manifest.csv", tmp_path / "shards", size=32)
    assert stats["written"] == {"test": 2} and stats["n_failures"] == 0

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_mod.main(
            [
                "--from-manifest",
                str(frozen_csv),
                "--expected-sha256",
                "0" * 64,
                "--cache-dir",
                str(cache),
                "--out",
                str(tmp_path / "bad_out"),
            ]
        )


def test_frozen_test_v2_single_shot_report_and_val_lock():
    folder = ROOT / "benchmarks/frozen-test-v2"

    # Decoding choice must be locked strictly from PMC val (963 images).
    val_sel = json.loads((folder / "val_selection.json").read_text(encoding="utf-8"))
    assert val_sel["split"] == "pmc_val"
    assert val_sel["n"] == 963
    variants = val_sel["variants"]
    assert len(variants) == 10
    expected_best = max(variants, key=lambda k: variants[k]["exact"])
    assert expected_best == "AC-ensemble-beam5-ttapad"
    assert val_sel["best_variant"] == expected_best

    # Score-once rule: only one report/prediction set may exist in benchmarks/frozen-test-v2/.
    md_reports = sorted(folder.glob("*-report.md"))
    json_reports = sorted(folder.glob("*-report.json"))
    pred_csvs = sorted(folder.glob("*-preds.csv"))
    assert [p.name for p in md_reports] == [f"{expected_best}-report.md"]
    assert [p.name for p in json_reports] == [f"{expected_best}-report.json"]
    assert [p.name for p in pred_csvs] == [f"{expected_best}-preds.csv"]

    pred_bytes = pred_csvs[0].read_bytes()
    assert b"\r\n" not in pred_bytes
    with open(pred_csvs[0], encoding="utf-8", newline="") as f:
        pred_rows = list(csv.DictReader(f))
    assert len(pred_rows) == 300

    rep = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert rep["input"] == f"{expected_best}-preds.csv"
    assert rep["n_scored"] == 300
    groups = {g["subset"]: g for g in rep["groups"]}
    assert groups["ALL"]["n"] == 300
    assert groups["ACS"]["n"] == 100
    assert groups["UOB"]["n"] == 100
    assert groups["USPTO"]["n"] == 100
    assert groups["has_stereo"]["n"] == 57
    assert groups["no_stereo"]["n"] == 243
    assert len(groups["ALL"]["exact_ci95"]) == 2
    assert len(groups["has_stereo"]["exact_ci95"]) == 2
    assert len(groups["no_stereo"]["exact_ci95"]) == 2

    md = md_reports[0].read_text(encoding="utf-8")
    for label in ("**ALL**", "ACS", "UOB", "USPTO", "has_stereo", "no_stereo"):
        assert f"| {label} |" in md
