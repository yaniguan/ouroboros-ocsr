"""Quantize an OCSR checkpoint (int8 dynamic decoder or fp16) and compare against fp32.

    python -m lab.quantize checkpoints/ouroboros-A.pt --method int8-dynamic \
        --shards real/pmc --split val --report benchmarks/quantize-cpu/report.json
    python -m lab.quantize checkpoints/ouroboros-A.pt --method fp16 --out runs/A-fp16.pt

Methods:
  none          keep full fp32 weights (exports steerable encoder to pure PyTorch)
  int8-dynamic  dynamic int8 quantization on decoder Linear layers (CPU-friendly)
  fp16          cast encoder + decoder parameters and buffers to float16
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.eval.metrics import aggregate, score_pair
from ouroboros.eval.standardize import standardize
from ouroboros.hf import EXPORTED_ENCODER_TYPES, inference_state
from ouroboros.model import OCSRModel, build_model, load_model_state

METHODS = ("none", "fp16", "int8-dynamic")


def load_fp32_model(ckpt_path: str | Path, device: str | torch.device = "cpu"):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tok = SmilesTokenizer(state["vocab"])
    model = build_model(state["config"], tok)
    load_model_state(model, state["model"])
    if hasattr(model.encoder, "export"):
        model.encoder = model.encoder.export()
    model.to(device).eval()
    return model, state


def quantize_model(model: OCSRModel, method: str) -> OCSRModel:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    out = copy.deepcopy(model).eval()
    if hasattr(out.encoder, "export"):
        out.encoder = out.encoder.export()
    if method == "int8-dynamic":
        out.decoder = torch.ao.quantization.quantize_dynamic(
            out.decoder, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif method == "fp16":
        out = out.half()
    return out


def input_dtype_for(method: str) -> torch.dtype:
    return torch.float16 if method == "fp16" else torch.float32


@torch.inference_mode()
def _decode_shards(
    model: OCSRModel,
    ds: ShardDataset,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 16,
) -> tuple[list[str], float]:
    # Warm up one tiny forward pass so timing reflects steady-state decoding.
    warm = ds[0]["image"][None].to(device=device, dtype=dtype)
    _ = model.generate(warm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    preds: list[str] = []
    t0 = time.perf_counter()
    for start in range(0, len(ds), batch_size):
        end = min(len(ds), start + batch_size)
        imgs = torch.stack([ds[i]["image"] for i in range(start, end)]).to(
            device=device, dtype=dtype
        )
        preds.extend(model.generate(imgs))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def compare_on_shards(
    ckpt_path: str | Path,
    shards_dir: str | Path,
    *,
    split: str = "val",
    method: str = "int8-dynamic",
    batch_size: int = 16,
    max_samples: int | None = None,
    device: str | torch.device = "cpu",
    checkpoint_label: str | None = None,
) -> dict:
    """Compare quantized greedy decoding against fp32 on the same split."""
    dev = torch.device(device)
    if method == "int8-dynamic" and dev.type != "cpu":
        raise ValueError("int8-dynamic decoder quantization runs on CPU only")
    fp32_model, state = load_fp32_model(ckpt_path, dev)
    quant_model = quantize_model(fp32_model, method).to(dev)
    img_size = state.get("config", {}).get("data", {}).get("image_size", 384)
    with ShardDataset(shards_dir, split, image_size=img_size, max_samples=max_samples) as ds:
        if len(ds) == 0:
            raise ValueError(f"no samples in {shards_dir}:{split}")

        golds = [ds.meta(i)["smiles"] for i in range(len(ds))]
        fp32_preds, fp32_s = _decode_shards(
            fp32_model, ds, device=dev, dtype=torch.float32, batch_size=batch_size
        )
        q_preds, q_s = _decode_shards(
            quant_model,
            ds,
            device=dev,
            dtype=input_dtype_for(method),
            batch_size=batch_size,
        )
        n = len(ds)

    fp32_agg = aggregate([score_pair(p, g) for p, g in zip(fp32_preds, golds, strict=True)])
    q_agg = aggregate([score_pair(p, g) for p, g in zip(q_preds, golds, strict=True)])
    same_raw = sum(a == b for a, b in zip(fp32_preds, q_preds, strict=True)) / n
    same_std = (
        sum(
            standardize(a).smiles == standardize(b).smiles
            for a, b in zip(fp32_preds, q_preds, strict=True)
        )
        / n
    )
    return {
        "checkpoint": checkpoint_label or Path(ckpt_path).name,
        "split": split,
        "n": n,
        "device": str(dev),
        "method": method,
        "fp32": {
            "exact": fp32_agg["exact"],
            "exact_flat": fp32_agg["exact_flat"],
            "valid": fp32_agg["valid"],
            "tanimoto": fp32_agg["tanimoto"],
            "seconds": round(fp32_s, 3),
            "ms_per_image": round(1000.0 * fp32_s / n, 2),
            "img_per_s": round(n / max(fp32_s, 1e-9), 2),
        },
        "quantized": {
            "exact": q_agg["exact"],
            "exact_flat": q_agg["exact_flat"],
            "valid": q_agg["valid"],
            "tanimoto": q_agg["tanimoto"],
            "seconds": round(q_s, 3),
            "ms_per_image": round(1000.0 * q_s / n, 2),
            "img_per_s": round(n / max(q_s, 1e-9), 2),
        },
        "exact_drop": round(fp32_agg["exact"] - q_agg["exact"], 6),
        "raw_smiles_agreement": round(same_raw, 6),
        "canonical_smiles_agreement": round(same_std, 6),
        "speedup": round(fp32_s / max(q_s, 1e-9), 3),
    }


def save_checkpoint_variant(
    ckpt_path: str | Path, out_path: str | Path, *, method: str
) -> dict:
    """Save an exported fp32/fp16 checkpoint or an int8-dynamic recipe bundle."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    model, state = load_fp32_model(ckpt_path, "cpu")
    cfg = {
        "model": copy.deepcopy(state["config"]["model"]),
        "data": {"image_size": state["config"].get("data", {}).get("image_size", 384)},
    }
    enc_type = cfg["model"]["encoder"]["type"]
    if enc_type in EXPORTED_ENCODER_TYPES:
        cfg["model"]["encoder"]["type"] = EXPORTED_ENCODER_TYPES[enc_type]
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if method == "fp16":
        model = model.half()
        payload = {
            "config": cfg,
            "vocab": list(state["vocab"]),
            "model": inference_state(model.state_dict()),
            "quantization": {"method": "fp16"},
        }
    else:
        payload = {
            "config": cfg,
            "vocab": list(state["vocab"]),
            "model": inference_state(model.state_dict()),
            "quantization": {
                "method": method,
                "apply_on_load": method == "int8-dynamic",
            },
        }
    torch.save(payload, dst)
    return {"saved": str(dst), "method": method, "bytes": dst.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="input .pt checkpoint")
    ap.add_argument("--method", choices=METHODS, default="int8-dynamic")
    ap.add_argument("--out", help="optional output .pt path")
    ap.add_argument("--shards", help="optional shards directory for validation check")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="device for shard comparison (default: cpu)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--checkpoint-label", default=None)
    ap.add_argument("--report", help="write JSON validation comparison here")
    a = ap.parse_args(argv)

    if a.threads:
        torch.set_num_threads(a.threads)

    out_info = None
    if a.out:
        out_info = save_checkpoint_variant(a.checkpoint, a.out, method=a.method)

    if a.shards:
        rep = compare_on_shards(
            a.checkpoint,
            a.shards,
            split=a.split,
            method=a.method,
            batch_size=a.batch_size,
            max_samples=a.max_samples,
            device=a.device,
            checkpoint_label=a.checkpoint_label,
        )
        if out_info:
            rep["artifact"] = out_info
        text = json.dumps(rep, indent=2) + "\n"
        if a.report:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0

    if out_info is None:
        # Validate load + quantization in memory when neither --out nor --shards is given.
        m, _ = load_fp32_model(a.checkpoint, "cpu")
        _ = quantize_model(m, a.method)
        out_info = {"checkpoint": Path(a.checkpoint).name, "method": a.method, "verified": True}
    print(json.dumps(out_info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
