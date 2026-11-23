"""Measure CPU latency across single-arm greedy vs ensemble + beam + TTA selective-gate modes.

    python scripts/bench_operating_modes.py --shards real/subset100_uob \
        --out benchmarks/cpu-api/operating_modes.json

Aligns latency numbers directly with decoding modes so users see the exact CPU cost of:
  1. Single-arm A (int8, greedy)
  2. Single-arm C-pure (int8, greedy, zero escnn)
  3. A + C-pure ensemble (int8, beam 5, no TTA)
  4. A + C-pure ensemble (int8, beam 5 + pad TTA + selective confidence gate)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.decode.confidence import confidence
from ouroboros.decode.ensemble import predict_batch
from ouroboros.infer import load_checkpoint


def _quantize_int8(model):
    model.decoder = torch.ao.quantization.quantize_dynamic(
        model.decoder, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model


def _time_mode(fn, images: list[torch.Tensor]) -> dict:
    # 1 warmup image
    with torch.inference_mode():
        fn(images[0])
    wall_ms = []
    cpu_ms = []
    for x in images:
        t0 = time.perf_counter()
        c0 = time.process_time()
        with torch.inference_mode():
            fn(x)
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        cpu_ms.append((time.process_time() - c0) * 1000.0)
    w = np.asarray(wall_ms, dtype=np.float64)
    c = np.asarray(cpu_ms, dtype=np.float64)
    return {
        "wall_ms": {
            "p50": round(float(np.percentile(w, 50)), 1),
            "p95": round(float(np.percentile(w, 95)), 1),
            "mean": round(float(w.mean()), 1),
        },
        "cpu_ms": {
            "p50": round(float(np.percentile(c, 50)), 1),
            "p95": round(float(np.percentile(c, 95)), 1),
            "mean": round(float(c.mean()), 1),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", default="real/subset100_uob")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ckpt-a", type=Path, default=Path("checkpoints/ouroboros-A.pt"))
    ap.add_argument("--ckpt-c", type=Path, default=Path("checkpoints/ouroboros-C-pure.pt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    with ShardDataset(args.shards, args.split, image_size=384, max_samples=args.n) as ds:
        samples = [ds[i]["image"][None] for i in range(len(ds))]

    mod_a, _ = load_checkpoint(args.ckpt_a, device="cpu", apply_int8=False)
    mod_c, _ = load_checkpoint(args.ckpt_c, device="cpu", apply_int8=False)
    mod_a = _quantize_int8(mod_a.eval())
    mod_c = _quantize_int8(mod_c.eval())

    modes = {}
    print("[bench_operating_modes] timing A_int8_greedy...", flush=True)
    modes["A_int8_greedy"] = {
        "description": "Single arm A, int8 decoder, greedy (1 encoder pass)",
        "encoder_passes": 1,
        "beam": 1,
        "tta": [],
        **_time_mode(lambda x: mod_a.generate(x), samples),
    }

    print("[bench_operating_modes] timing C_pure_int8_greedy...", flush=True)
    modes["C_pure_int8_greedy"] = {
        "description": "Single arm C-pure (C8 steerable, zero escnn), int8 decoder, greedy",
        "encoder_passes": 1,
        "beam": 1,
        "tta": [],
        **_time_mode(lambda x: mod_c.generate(x), samples),
    }

    print("[bench_operating_modes] timing AC_int8_beam5...", flush=True)
    modes["AC_int8_beam5"] = {
        "description": "A + C-pure ensemble, int8 decoder, beam 5 + RDKit validity rerank",
        "encoder_passes": 2,
        "beam": 5,
        "tta": [],
        **_time_mode(lambda x: predict_batch([mod_a, mod_c], x, 5, [], True), samples),
    }

    print("[bench_operating_modes] timing AC_int8_beam5_ttapad_gated...", flush=True)

    def _gated(x: torch.Tensor):
        src: list = []
        res = predict_batch([mod_a, mod_c], x, 5, ["pad"], True, diagnostics=src)
        pred, ranked = res[0]
        confidence(pred, ranked, src[0], min_margin=0.05, max_disagreement=0.0)

    modes["AC_int8_beam5_ttapad_gated"] = {
        "description": "A + C-pure ensemble, int8 decoder, beam 5 + pad TTA + selective gate",
        "encoder_passes": 4,
        "beam": 5,
        "tta": ["pad"],
        "gate": {"min_margin": 0.05, "max_disagreement": 0.0},
        **_time_mode(_gated, samples),
    }

    base_p50 = modes["A_int8_greedy"]["wall_ms"]["p50"]
    for m in modes.values():
        m["relative_wall_p50"] = round(m["wall_ms"]["p50"] / max(base_p50, 1e-6), 2)

    out_obj = {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": int(torch.get_num_interop_threads() or 1),
        "threads": args.threads,
        "n_images": len(samples),
        "dataset": "Keylab/OCSR-Benchmarks UOB prefix (seed=0 dev-test)",
        "frozen_test_v2_reference_accuracy": {
            "A_C_beam5_ttapad_ungated_exact": 0.4667,
            "A_C_beam5_ttapad_selective_precision": 0.9778,
            "selective_coverage_by_subset": {
                "ALL": 0.30,
                "UOB": 0.70,
                "ACS": 0.10,
                "USPTO": 0.10,
            },
        },
        "modes": modes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["wall_ms"] for k, v in modes.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
