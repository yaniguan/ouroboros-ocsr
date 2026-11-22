"""Regression tests for ouroboros.infer and lab/* audit fixes."""

from __future__ import annotations

import copy
import io
import json
import tarfile

import pytest
import torch
from PIL import Image

from lab import edge_plan as ep
from lab import grpo_verifier as gv
from lab import molecule_context as mc
from lab import quantize as lq
from ouroboros.data.real import ingest
from ouroboros.infer import InputError, OCSRPredictor, load_checkpoint, open_image
from ouroboros.ui.backend import _build_small_probe_model


def _png_bytes(w: int = 48, h: int = 48, fill: int = 240) -> bytes:
    img = Image.new("L", (w, h), fill)
    img.putpixel((w // 2, h // 2), 10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("method", ["none", "int8-dynamic", "fp16"])
def test_steerable_quantize_saved_checkpoint_reloads_cleanly(tmp_path, method):
    model, cfg = _build_small_probe_model("steerable", size=32, order=4)
    src_ckpt = tmp_path / "steerable-src.pt"
    torch.save(
        {"config": cfg, "vocab": model.tokenizer.itos, "model": model.state_dict()}, src_ckpt
    )
    out_ckpt = tmp_path / f"steerable-{method}.pt"
    info = lq.save_checkpoint_variant(src_ckpt, out_ckpt, method=method)
    assert info["method"] == method

    # Exported checkpoints contain only plain config/vocab and tensors (weights_only=True safe).
    reloaded, out_cfg = load_checkpoint(out_ckpt, allow_unsafe_pickle=False)
    assert out_cfg["model"]["encoder"]["type"] == "exported_steerable"
    pred = OCSRPredictor(
        model=reloaded,
        image_size=out_cfg["data"]["image_size"],
        int8=(method == "int8-dynamic"),
    )
    res = pred.predict(_png_bytes(32, 32))
    assert "smiles" in res and 0.0 <= res["confidence"] <= 1.0


def test_infer_pil_pixel_limit_no_inplace_mutation_and_http_413(monkeypatch):
    model, cfg = _build_small_probe_model("baseline", size=64)
    orig_linear = type(model.decoder.head)
    pred = OCSRPredictor(model=model, image_size=cfg["data"]["image_size"], int8=True)
    # Caller's model must remain fp32 nn.Linear while predictor uses its own quantized copy.
    assert type(model.decoder.head) is orig_linear

    monkeypatch.setattr("ouroboros.infer.MAX_PIXELS", 100)
    with pytest.raises(InputError, match="pixels"):
        open_image(Image.new("L", (20, 20), 255))

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ouroboros.infer import create_app

    monkeypatch.setattr("ouroboros.infer.MAX_PIXELS", 25_000_000)
    monkeypatch.setattr("ouroboros.infer.MAX_BYTES", 32)
    client = TestClient(create_app(pred))
    resp = client.post("/predict", content=_png_bytes(64, 64))
    assert resp.status_code == 413


def test_grpo_verifier_restores_max_batch_size_and_preserves_caller_model():
    model, cfg = _build_small_probe_model("baseline", size=64)
    orig_head = type(model.decoder.head)
    verifier = gv.OCSRVerifier(
        model=model,
        device="cpu",
        dtype="fp32",
        int8_cpu=True,
        image_size=cfg["data"]["image_size"],
        max_batch_size=5,
    )
    assert type(model.decoder.head) is orig_head
    _ = gv.benchmark_batch_sizes(verifier, batch_sizes=[1, 2], repeats=1)
    assert verifier.max_batch_size == 5


def test_pseudo_label_manifest_from_unlabeled_shards_ingests_end_to_end(tmp_path):
    unlabeled_dir = tmp_path / "unlabeled"
    unlabeled_dir.mkdir()
    with tarfile.open(unlabeled_dir / "train-000000.tar", "w") as tar:
        for idx, fig_id in enumerate(("fig/01", "fig/02")):
            png = _png_bytes(48, 48, 240 - idx * 5)
            meta = json.dumps({"id": fig_id, "source": "lab_figs"}).encode()
            for ext, blob in (("png", png), ("json", meta)):
                ti = tarfile.TarInfo(f"u{idx}.{ext}")
                ti.size = len(blob)
                tar.addfile(ti, io.BytesIO(blob))

    preds_csv = tmp_path / "teacher.csv"
    preds_csv.write_text(
        "id,gold,pred,subset,accepted,rejection_reason,margin,disagreement\n"
        "fig/01,,CCO,lab_figs,True,,0.25,0.0\n"
        "fig/02,,C[C@H](O)Cl,lab_figs,False,low_margin,0.01,0.0\n",
        encoding="utf-8",
    )
    manifest_out = tmp_path / "pseudo" / "manifest.csv"
    jsonl_out = tmp_path / "pseudo" / "context.jsonl"
    rc = mc.main(
        [
            str(preds_csv),
            "--out",
            str(jsonl_out),
            "--manifest-out",
            str(manifest_out),
            "--source-shards",
            f"{unlabeled_dir}:train",
        ]
    )
    assert rc == 0
    stats = ingest(manifest_out, tmp_path / "pseudo" / "shards", size=64)
    assert stats["n_failures"] == 0
    assert stats["written"] == {"train": 1}


def test_edge_plan_validation_allows_pytest_parent_and_rejects_bool_or_missing_keys():
    cfg = copy.deepcopy(ep.load_config(ep.DEFAULT_CONFIG))
    cfg["selection"]["shards"] = "/tmp/pytest-of-runner/latest/pmc"
    cfg["selection"]["split"] = "val"
    assert ep.validate(cfg) is cfg

    for mut in (
        lambda c: c["teacher"].__setitem__("beam", True),
        lambda c: c["pseudo_labels"].__setitem__("min_margin", False),
        lambda c: c["student"].pop("config"),
        lambda c: c["selection"].pop("shards"),
    ):
        bad = copy.deepcopy(cfg)
        mut(bad)
        with pytest.raises(ep.PlanError):
            ep.validate(bad)
