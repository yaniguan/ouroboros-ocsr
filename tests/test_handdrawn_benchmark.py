"""Tests for the DECIMER Hand-drawn evaluation manifest, fetcher, leakage receipt, and reports."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from ouroboros.data.real import (
    DECIMER_HDM_SOURCE,
    EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256,
    HANDDRAWN_DECIMER_MANIFEST,
    HANDDRAWN_DECIMER_N,
    HANDDRAWN_DECIMER_SEED,
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


def _png_bytes(shade: int = 235) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (shade, shade, shade)).save(buf, format="PNG")
    return buf.getvalue()


def test_handdrawn_decimer_manifest_and_reports():
    manifest = ROOT / HANDDRAWN_DECIMER_MANIFEST
    assert manifest.is_file()
    raw = manifest.read_bytes()
    assert b"\r\n" not in raw, "Committed hand-drawn manifest must use LF line endings"
    assert len(EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256) == 64
    assert file_sha256(manifest) == EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256
    assert HANDDRAWN_DECIMER_SEED == 2026

    rows = read_real_manifest(manifest)
    assert len(rows) == HANDDRAWN_DECIMER_N
    assert all(r["source"] == DECIMER_HDM_SOURCE for r in rows)
    assert all(r["split"] == "test" for r in rows)

    keys = [leakage_key(standardize(r["smiles"])) for r in rows]
    assert all(k is not None for k in keys)
    assert len({k[:14] for k in keys if k is not None}) == HANDDRAWN_DECIMER_N

    folder = manifest.parent
    image_files = [
        p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}
    ]
    assert image_files == []

    receipt = json.loads((folder / "leakage.json").read_text(encoding="utf-8"))
    assert len(receipt) >= 4
    assert all(
        r["n_eval"] == HANDDRAWN_DECIMER_N
        and r["overlap_inchikey"] == 0
        and r["overlap_connectivity"] == 0
        for r in receipt
    )

    expected_models = [
        "A-released",
        "B-released",
        "C-released",
        "A-selected",
        "C-selected",
        "AC-ensemble-beam5",
        "AC-ensemble-beam5-ttapad",
    ]
    for model_name in expected_models:
        md = (folder / f"{model_name}-report.md").read_text(encoding="utf-8")
        assert "| **ALL** | 200 |" in md
        assert "| has_stereo |" in md
        assert "| no_stereo |" in md
        rep = json.loads((folder / f"{model_name}-report.json").read_text(encoding="utf-8"))
        assert rep["input"] == f"{model_name}-preds.csv"
        assert rep["n_scored"] == HANDDRAWN_DECIMER_N
        groups = {g["subset"]: g for g in rep["groups"]}
        assert groups["has_stereo"]["n"] + groups["no_stereo"]["n"] == HANDDRAWN_DECIMER_N
        assert len(groups["ALL"]["exact_ci95"]) == 2


def test_fetch_handdrawn_subset_offline_roundtrip(tmp_path, monkeypatch):
    mod = _load_script("fetch_handdrawn_subset")
    cache = tmp_path / "cache"
    cache.mkdir()

    tsv = cache / "DECIMER_HDM_Dataset_SMILES.tsv"
    tsv.write_text(
        "IDs\tSMILES\n"
        "HD_1\tCCO\n"
        "HD_2\tC[C@H](O)C(=O)O\n"
        "HD_3\tC[C@@H](O)C(=O)O\n"
        "HD_4\tc1ccccc1\n"
        "HD_5\tINVALID_SMILES(\n",
        encoding="utf-8",
    )

    tgz = cache / "DECIMER_HDM_Dataset_Images.tar.gz"
    png = _png_bytes()
    with tarfile.open(tgz, "w:gz") as tf:
        for hid in ("HD_1", "HD_2", "HD_3", "HD_4"):
            info = tarfile.TarInfo(name=f"DECIMER_HDM_Dataset_Images/{hid}.png")
            info.size = len(png)
            tf.addfile(info, io.BytesIO(png))

    monkeypatch.setattr(mod, "TSV_MD5", None)
    monkeypatch.setattr(mod, "TSV_SHA256", None)
    monkeypatch.setattr(mod, "IMAGES_MD5", None)

    excl = tmp_path / "excl.csv"
    excl.write_text("smiles\nCCO\n", encoding="utf-8")

    # HD_1 excluded (CCO); HD_2 and HD_3 share lactic-acid connectivity block; HD_4 is benzene.
    out = tmp_path / "out"
    assert (
        mod.main(
            [
                "--n",
                "2",
                "--seed",
                "7",
                "--exclude-manifest",
                str(excl),
                "--cache",
                str(cache),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    manifest = out / "manifest.csv"
    with open(manifest, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    ids = {r["id"] for r in rows}
    assert "HD_1" not in ids
    assert "HD_4" in ids
    assert len(ids & {"HD_2", "HD_3"}) == 1

    # Materialize into a fresh directory with SHA-256 check, then ingest.
    mat = tmp_path / "materialized"
    sha = file_sha256(manifest)
    assert (
        mod.main(
            [
                "--from-manifest",
                str(manifest),
                "--expected-sha256",
                sha,
                "--cache",
                str(cache),
                "--out",
                str(mat),
            ]
        )
        == 0
    )
    stats = ingest(mat / "manifest.csv", tmp_path / "shards", size=64)
    assert stats["written"] == {"test": 2}

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mod.main(
            [
                "--from-manifest",
                str(manifest),
                "--expected-sha256",
                "0" * 64,
                "--cache",
                str(cache),
                "--out",
                str(tmp_path / "bad"),
            ]
        )
