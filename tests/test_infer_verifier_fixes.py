"""Regression tests for Issue #121: fp16/int8 dtypes and RotationLab pure-PyTorch exports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from lab.grpo_verifier import OCSRVerifier
from lab.quantize import save_checkpoint_variant
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.demo.rotation import EXAMPLES, RotationLab, tiny_config
from ouroboros.hf import CONFIG_FILE, FORMAT, VOCAB_FILE, inference_state, save_weights
from ouroboros.infer import OCSRPredictor, load_checkpoint
from ouroboros.model import build_model
from scripts.export_hf_release import symmetry_group


def _tiny_ckpt(tmp_path: Path) -> Path:
    cfg = tiny_config(4)
    tok = SmilesTokenizer.build(EXAMPLES.values())
    torch.manual_seed(0)
    model = build_model(cfg, tok).eval()
    ckpt = tmp_path / "tiny.pt"
    torch.save(
        {"model": model.state_dict(), "config": cfg, "vocab": tok.itos, "step": 10},
        ckpt,
    )
    return ckpt


def test_ocsr_verifier_handles_fp16_and_int8_checkpoints_on_cpu(tmp_path: Path) -> None:
    src = _tiny_ckpt(tmp_path)
    fp16_ckpt = tmp_path / "tiny_fp16.pt"
    int8_ckpt = tmp_path / "tiny_int8.pt"
    save_checkpoint_variant(src, fp16_ckpt, method="fp16")
    save_checkpoint_variant(src, int8_ckpt, method="int8-dynamic")

    img = Image.new("L", (128, 128), color=255)

    # Default auto dtype on CPU with fp16 checkpoint keeps fp16 weights & fp16 inputs
    v_auto = OCSRVerifier(fp16_ckpt, device="cpu", dtype="auto")
    assert v_auto.input_dtype == torch.float16
    rows_auto = v_auto.verify_batch([img])
    assert len(rows_auto) == 1 and "status" in rows_auto[0]

    # Explicit fp32 casts fp16 weights to float32 so verify_batch never mismatches
    v_fp32 = OCSRVerifier(fp16_ckpt, device="cpu", dtype="fp32")
    assert v_fp32.input_dtype == torch.float32
    rows_fp32 = v_fp32.verify_batch([img])
    assert len(rows_fp32) == 1

    # fp32 + int8_cpu on an fp16 checkpoint casts to float32 then quantizes decoder
    v_int8 = OCSRVerifier(fp16_ckpt, device="cpu", dtype="fp32", int8_cpu=True)
    has_qint8 = any(
        isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in v_int8.model.decoder.modules()
    )
    assert has_qint8
    assert len(v_int8.verify_batch([img])) == 1


def test_ocsr_predictor_respects_fp32_flag_name_and_hf_folder(tmp_path: Path) -> None:
    src = _tiny_ckpt(tmp_path)
    fp16_ckpt = tmp_path / "tiny_fp16.pt"
    int8_ckpt = tmp_path / "tiny_int8.pt"
    save_checkpoint_variant(src, fp16_ckpt, method="fp16")
    save_checkpoint_variant(src, int8_ckpt, method="int8-dynamic")

    pred_fp32 = OCSRPredictor(int8_ckpt, int8=False)
    assert not any(
        isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
        for m in pred_fp32.model.decoder.modules()
    )
    assert pred_fp32.name == "tiny_int8.pt"

    pred_fp16 = OCSRPredictor(fp16_ckpt, int8=True)
    assert pred_fp16.name == "tiny_fp16.pt"

    # Exported HF directory loads cleanly via load_checkpoint and RotationLab
    raw_state = torch.load(src, map_location="cpu", weights_only=True)
    hf_dir = tmp_path / "hf_export"
    hf_dir.mkdir()
    save_weights(inference_state(raw_state["model"]), hf_dir, "model")
    (hf_dir / VOCAB_FILE).write_text(json.dumps(raw_state["vocab"]), encoding="utf-8")
    (hf_dir / CONFIG_FILE).write_text(
        json.dumps(
            {
                "format": FORMAT,
                "vocab_size": len(raw_state["vocab"]),
                "model": raw_state["config"]["model"],
                "data": raw_state["config"]["data"],
            }
        ),
        encoding="utf-8",
    )
    loaded_model, loaded_cfg = load_checkpoint(hf_dir)
    assert loaded_cfg["data"]["image_size"] == 128
    assert isinstance(loaded_model.generate(torch.zeros(1, 1, 128, 128))[0], str)
    hf_lab = RotationLab.from_checkpoint(hf_dir)
    assert hf_lab.order == 4


def test_export_hf_release_canonicalized_symmetry_label() -> None:
    enc = SimpleNamespace(c=SimpleNamespace(N=8))
    label = symmetry_group({"type": "canonicalized", "N": 8}, enc)
    assert "canonicalization" in label


def test_rotation_lab_supports_exported_steerable_and_predict(tmp_path: Path) -> None:
    src = _tiny_ckpt(tmp_path)
    exported_ckpt = tmp_path / "exported_c4.pt"
    save_checkpoint_variant(src, exported_ckpt, method="none")

    lab_orig = RotationLab.from_checkpoint(src, device="cpu")
    lab_exp = RotationLab(ckpt_path=exported_ckpt, device="cpu")
    assert lab_exp.order == 4

    res_orig = lab_orig.evaluate("CCO", angle=90.0)
    res_exp = lab_exp.evaluate("CCO", angle=90.0)
    assert res_exp["on_group"] is True
    assert res_exp["feature_error"] is not None and res_exp["feature_error"] < 1e-5
    assert abs(res_exp["feature_error"] - res_orig["feature_error"]) < 1e-6

    pred = lab_exp.predict(Image.new("L", (128, 128), color=255))
    assert "smiles" in pred and "accepted" in pred
