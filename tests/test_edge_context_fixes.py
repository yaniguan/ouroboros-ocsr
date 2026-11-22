"""Regression tests for Issue #122: edge_plan eval_keys/quantize wiring and context PNG/schema."""

from __future__ import annotations

import copy
import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from lab import edge_plan as ep
from lab import molecule_context as mc
from scripts.predict_real import _row_id


def _write_unlabeled_shard(root: Path, split: str, items: list[tuple[str, tuple[int, int, int]]]):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / f"{split}-000000.tar", "w") as tf:
        for i, (rid, rgb) in enumerate(items):
            key = f"{i:06d}"
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), color=rgb).save(buf, format="PNG")
            png_bytes = buf.getvalue()
            json_bytes = json.dumps({"id": rid, "source": "PMC"}).encode("utf-8")

            for ext, payload in (("png", png_bytes), ("json", json_bytes)):
                ti = tarfile.TarInfo(f"{key}.{ext}")
                ti.size = len(payload)
                tf.addfile(ti, io.BytesIO(payload))


def test_edge_plan_wires_eval_keys_val_leakage_and_quantized_ckpt() -> None:
    cfg = ep.load_config(ep.DEFAULT_CONFIG)
    steps = {s["step"]: s["cmd"] for s in ep.plan(cfg)}

    assert "val=real/pmc:val" in steps[4]
    assert "--dump-eval-keys runs/ocsr-edge-v0/eval_keys.txt" in steps[4]
    assert "--exclude-keys runs/ocsr-edge-v0/eval_keys.txt" in steps[5]
    assert "--out runs/ocsr-edge-v0/student_quantized.pt" in steps[7]
    assert "--ckpt runs/ocsr-edge-v0/student_quantized.pt" in steps[8]

    # Scratch-training branch (no init checkpoint) also overrides data.exclude_keys
    cfg_no_init = copy.deepcopy(cfg)
    cfg_no_init["student"].pop("init", None)
    steps_no_init = {s["step"]: s["cmd"] for s in ep.plan(cfg_no_init)}
    assert "data.exclude_keys=runs/ocsr-edge-v0/eval_keys.txt" in steps_no_init[5]


def test_from_json_dict_enforces_strict_boolean_usable_invariant() -> None:
    valid_ctx = mc.from_prediction({"id": "m1", "pred": "CCO"})
    base = json.loads(valid_ctx.to_json())
    assert mc.from_json_dict(base).usable is True

    bad_str = {**base, "usable": "false"}
    with pytest.raises(ValueError, match="bool"):
        mc.from_json_dict(bad_str)

    bad_mismatch = {**base, "status": "review", "usable": True}
    with pytest.raises(ValueError, match="disagrees"):
        mc.from_json_dict(bad_mismatch)

    bad_empty_smi = {**base, "smiles": ""}
    with pytest.raises(ValueError, match="smiles"):
        mc.from_json_dict(bad_empty_smi)


def test_extract_shard_pngs_disambiguates_colliding_ids_and_empty_id(tmp_path: Path) -> None:
    shard_root = tmp_path / "shards"
    _write_unlabeled_shard(
        shard_root,
        "train",
        [
            ("PMC123/fig 1", (255, 0, 0)),
            ("PMC123:fig_1", (0, 255, 0)),
            ("", (0, 0, 255)),
        ],
    )
    assert _row_id({"id": ""}, 2) == 2

    ctxs = [
        mc.from_prediction({"id": "PMC123/fig 1", "pred": "CCO"}),
        mc.from_prediction({"id": "PMC123:fig_1", "pred": "CC(=O)O"}),
        mc.from_prediction({"id": str(_row_id({"id": ""}, 2)), "pred": "c1ccccc1"}),
    ]
    manifest = tmp_path / "pseudo" / "manifest.csv"
    n = mc.write_pseudo_manifest(ctxs, manifest, source_shards=f"{shard_root}:train")
    assert n == 3

    with open(manifest, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    img_paths = [r["image"] for r in rows]
    assert len(set(img_paths)) == 3  # no silent overwrite on sanitized ID collision

    pixels = []
    for rel in img_paths:
        with Image.open(manifest.parent / rel) as im:
            pixels.append(im.convert("RGB").getpixel((0, 0)))
    assert pixels == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
