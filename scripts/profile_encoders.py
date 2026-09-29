"""Parameters, FLOPs and throughput of encoder configs; search param- and FLOP-matched steerable
configs relative to the baseline (Phase 3 DoD). Writes benchmarks/encoders/profile.json.

FLOPs are counted with ``torch.utils.flop_counter.FlopCounterMode`` for one 384x384 image,
encoder only (the decoder is shared and identical across arms), in eval mode on the exported
(pure PyTorch) steerable model, so escnn's filter expansion is not counted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from ouroboros.encoder.baseline import BaselineConfig, BaselineEncoder
from ouroboros.encoder.steerable import SteerableConfig, SteerableEncoder

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "encoders"
BASE_FIELDS = [8, 16, 32, 64]


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def flops(m, size: int) -> int:
    x = torch.zeros(1, 1, size, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return int(fc.get_total_flops())


def throughput(m, size: int, batch: int, reps: int) -> float:
    x = torch.rand(batch, 1, size, size)
    with torch.no_grad():
        m(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(x)
    return batch * reps / (time.perf_counter() - t0)


def steerable(N: int, mult: float, size: int, stem_kernel: int = 7) -> SteerableEncoder:
    fields = [max(1, round(f * mult * 8 / N)) for f in BASE_FIELDS]  # same #channels for any N
    return SteerableEncoder(
        SteerableConfig(N=N, fields=fields, image_size=size, stem_kernel=stem_kernel)
    ).eval()


def profile(m, size, batch, reps, export=False) -> dict:
    run = m.export() if export else m
    return {
        "params": n_params(m),
        "params_trunk": n_params(m.trunk) if hasattr(m, "trunk") else None,
        "gflops": flops(run, size) / 1e9,
        "img_per_s_cpu": throughput(run, size, batch, reps),
    }


def search(N: int, target: float, key: str, size: int, lo=0.2, hi=4.0, iters=9) -> tuple:
    """Bisection on the width multiplier so that ``key`` (params or gflops) hits ``target``."""
    best = None
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        m = steerable(N, mid, size)
        val = n_params(m) if key == "params" else flops(m.export(), size) / 1e9
        rel = val / target - 1
        if best is None or abs(rel) < abs(best[2]):
            best = (mid, val, rel)
        if abs(rel) < 0.03:
            break
        lo, hi = (mid, hi) if val < target else (lo, mid)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--N", type=int, nargs="+", default=[8])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(0)
    base = BaselineEncoder(BaselineConfig(image_size=a.size)).eval()
    res = {"size": a.size, "baseline": profile(base, a.size, a.batch, a.reps), "steerable": {}}
    print("baseline", res["baseline"], flush=True)
    for N in a.N:
        out = {}
        for key, target in (
            ("params", res["baseline"]["params"]),
            ("gflops", res["baseline"]["gflops"]),
        ):
            mult, val, rel = search(N, target, key, a.size)
            m = steerable(N, mult, a.size)
            prof = profile(m, a.size, a.batch, a.reps, export=True)
            prof_train = throughput(m.train(), a.size, a.batch, 1)
            out[f"{key}_matched"] = {
                "width_mult": mult,
                "fields": m.c.fields,
                "rel_diff_vs_baseline": rel,
                **prof,
                "img_per_s_cpu_train_mode_escnn": prof_train,
            }
            print(N, key, out[f"{key}_matched"], flush=True)
        res["steerable"][f"C{N}"] = out
    res["date"] = time.strftime("%Y-%m-%d")
    res["note"] = "CPU throughput: 4-core dev VM, fp32, eval mode; GPU numbers come from Colab."
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "profile.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
