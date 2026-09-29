"""HF snapshot -> real manifests: column/split/source detection on the common HF layouts."""

import csv
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ouroboros.data.hf_manifest import (
    config_from_path,
    detect_columns,
    inspect_snapshot,
    split_from_path,
    write_manifests,
)
from ouroboros.data.loader import ShardDataset
from ouroboros.data.real import ingest


def _png(w=80, h=40, fmt="PNG") -> bytes:
    b = io.BytesIO()
    Image.new("L", (w, h), 200).save(b, format=fmt)
    return b.getvalue()


def test_split_and_config_from_path():
    from pathlib import Path

    assert split_from_path(Path("data/train-00000-of-00002.parquet")) == "train"
    assert split_from_path(Path("uspto/validation-00000-of-00001.parquet")) == "val"
    assert split_from_path(Path("test/metadata.jsonl")) == "test"
    assert split_from_path(Path("data/whatever.parquet")) is None
    assert config_from_path(Path("uspto/test-0.parquet")) == "uspto"
    assert config_from_path(Path("data/test-0.parquet")) is None


def test_detect_columns_prefers_parsable_smiles():
    rows = [
        {"file_name": f"{i}.png", "text": "a caption", "gt": "CCO", "dataset": "acs"}
        for i in range(5)
    ]
    col = detect_columns(rows)
    assert (col.image, col.smiles, col.source) == ("file_name", "gt", "dataset")
    assert detect_columns(rows, {"smiles": "text"}).smiles == "text"


def _hf_parquet(root):
    (root / "data").mkdir(parents=True)
    splits = {"train": ["CCO", "c1ccccc1O"], "test": ["C[C@H](N)C(=O)O", "CC(=O)[O-].[Na+]"]}
    for split, smis in splits.items():
        t = pa.table(
            {
                "image": [
                    {"bytes": _png(), "path": None},
                    {"bytes": _png(30, 90, "JPEG"), "path": None},
                ],
                "smiles": smis,
                "source": ["USPTO", "CLEF IP"],
            }
        )
        pq.write_table(t, root / "data" / f"{split}-00000-of-00001.parquet")


def _hf_imagefolder(root):
    (root / "acs" / "validation").mkdir(parents=True)
    with open(root / "acs" / "validation" / "metadata.jsonl", "w") as f:
        for i, smi in enumerate(["CCN", "C1CC1", ""]):
            (root / "acs" / "validation" / f"{i}.png").write_bytes(_png())
            f.write(json.dumps({"file_name": f"{i}.png", "SMILES": smi}) + "\n")


def test_parquet_layout_end_to_end(tmp_path):
    snap, out = tmp_path / "snap", tmp_path / "out"
    _hf_parquet(snap)
    rep = inspect_snapshot(snap)
    assert [t["detected"]["image"] for t in rep["tables"]] == ["image", "image"]
    assert rep["tables"][0]["source_column_values"] == {"USPTO": 1, "CLEF IP": 1}
    stats = write_manifests(snap, out)
    assert stats["rows"] == {
        "CLEF_IP:test": 1,
        "CLEF_IP:train": 1,
        "USPTO:test": 1,
        "USPTO:train": 1,
    }
    st = ingest(out / "manifests" / "USPTO.csv", tmp_path / "real" / "USPTO", size=64)
    assert st["written"] == {"train": 1, "test": 1} and st["n_failures"] == 0
    ds = ShardDataset(tmp_path / "real" / "USPTO", "test")
    assert ds.meta(0)["smiles"] == "C[C@H](N)C(=O)O" and ds.meta(0)["source"] == "USPTO"
    st = ingest(out / "manifests" / "CLEF_IP.csv", tmp_path / "real" / "CLEF_IP", size=64)
    assert ShardDataset(tmp_path / "real" / "CLEF_IP", "test").meta(0)["smiles"] == "CC(=O)O"


def test_imagefolder_layout_and_split_map(tmp_path):
    snap, out = tmp_path / "snap", tmp_path / "out"
    _hf_imagefolder(snap)
    stats = write_manifests(snap, out)
    assert stats["rows"] == {"acs:val": 2} and stats["skipped"] == {"no_smiles": 1}
    with open(out / "manifests" / "acs.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["image"].endswith("acs/validation/0.png") and rows[0]["id"] == "0.png"
    stats = write_manifests(snap, tmp_path / "out2", split_map={"val": "test"})
    assert stats["rows"] == {"acs:test": 2}
    st = ingest(tmp_path / "out2" / "manifests" / "acs.csv", tmp_path / "real" / "acs", size=64)
    assert st["written"] == {"test": 2}


def test_unknown_split_defaults_to_test(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "x.png").write_bytes(_png())
    with open(snap / "labels.csv", "w", newline="") as f:
        csv.writer(f).writerows([["img", "smi"], ["x.png", "CCO"]])
    stats = write_manifests(snap, tmp_path / "out", default_source="mine")
    assert stats["rows"] == {"mine:test": 1}


def test_pretty_printed_and_mapping_json(tmp_path):
    """Pretty-printed JSON (the Colab failure: '{' then a newline), filename->SMILES mappings,
    metadata objects without rows and broken files are all handled per file."""
    snap = tmp_path / "snap"
    (snap / "images").mkdir(parents=True)
    for n in ("a.png", "b.png", "c.png"):
        (snap / "images" / n).write_bytes(_png())
    recs = [{"file_name": "images/a.png", "smiles": "CCO", "split": "train"}]
    (snap / "train.json").write_text(json.dumps({"version": 1, "data": recs}, indent=2))
    (snap / "labels_test.json").write_text(
        json.dumps({"images/b.png": "c1ccccc1", "images/c.png": "CC(C)O"}, indent=2)
    )
    (snap / "info.json").write_text(json.dumps({"description": "x", "n": 3}, indent=2))
    (snap / "broken.json").write_text('{\n  "a": 1,\n')
    rep = inspect_snapshot(snap)
    by_file = {t["file"]: t for t in rep["tables"]}
    assert by_file["train.json"]["rows"] == 1 and by_file["labels_test.json"]["rows"] == 2
    assert by_file["info.json"]["rows"] == 0 and "json_top_level_keys" in by_file["info.json"]
    assert "error" in by_file["broken.json"]
    assert rep["layout"]["files_by_ext"] == {".json": 4, ".png": 3}
    stats = write_manifests(snap, tmp_path / "out", default_source="mine")
    assert stats["rows"] == {"mine:test": 2, "mine:train": 1}
