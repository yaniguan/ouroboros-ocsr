"""Fast offline tests for lab/quantize.py (int8-dynamic and fp16 quantization)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from lab import quantize as lq
from ouroboros.data.real import ingest
from ouroboros.ui.backend import _build_small_probe_model


def _tiny_ckpt_and_shards(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model, cfg = _build_small_probe_model("baseline", size=64)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"config": cfg, "vocab": model.tokenizer.itos, "model": model.state_dict()}, ckpt)

    img_dir = tmp_path / "raw"
    img_dir.mkdir()
    rows = ["image,smiles,source,split,id"]
    for idx, smi in enumerate(["CCO", "c1ccccc1", "CC(=O)O"]):
        img_path = img_dir / f"{idx}.png"
        Image.new("L", (64, 64), 245 - idx * 10).save(img_path)
        rows.append(f"{img_path.name},{smi},SYN,val,m{idx}")
    manifest = img_dir / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    shards = tmp_path / "shards"
    ingest(manifest, shards, size=64)
    return ckpt, shards


@pytest.mark.parametrize("method", ["none", "int8-dynamic", "fp16"])
def test_quantize_compare_and_cli(tmp_path, method):
    ckpt, shards = _tiny_ckpt_and_shards(tmp_path / method)
    out_pt = tmp_path / f"out-{method}.pt"
    report = tmp_path / f"rep-{method}.json"
    rc = lq.main(
        [
            str(ckpt),
            "--method",
            method,
            "--out",
            str(out_pt),
            "--shards",
            str(shards),
            "--split",
            "val",
            "--batch-size",
            "2",
            "--checkpoint-label",
            "tiny.pt",
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    assert out_pt.is_file()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["checkpoint"] == "tiny.pt"
    assert saved["n"] == 3
    assert saved["method"] == method
    assert "exact_drop" in saved
    if method == "none":
        assert saved["raw_smiles_agreement"] == 1.0
        assert saved["exact_drop"] == 0.0


def test_unknown_method_rejected(tmp_path):
    ckpt, _ = _tiny_ckpt_and_shards(tmp_path / "bad")
    m, _ = lq.load_fp32_model(ckpt)
    with pytest.raises(ValueError, match="unknown method"):
        lq.quantize_model(m, "int4")
