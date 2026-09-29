"""`.export()` check (Phase 3 DoD): exported pure-PyTorch encoder vs. the escnn model on rendered
test images (eval mode, fp32) — max relative error and inference speedup.
Writes benchmarks/encoders/export.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.model import build_encoder

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "encoders"


def bench(fn, x, reps):
    with torch.no_grad():
        fn(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
    return x.shape[0] * reps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="steerable")
    ap.add_argument("--fields", type=int, nargs=4, default=[7, 14, 27, 55])
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(0)
    enc = build_encoder({"type": a.type, "N": a.N, "fields": a.fields}, 512, a.size)
    ds = ShardDataset("data/full/shards", "test", image_size=a.size)
    x = torch.stack([ds[i]["image"] for i in range(a.batch)])
    enc.train()
    with torch.no_grad():  # non-trivial BN running statistics
        for i in range(3):
            enc(torch.stack([ds[a.batch * (i + 1) + j]["image"] for j in range(a.batch)]))
    enc.eval()
    t_train_mode = bench(lambda z: enc.train()(z), x, 1)
    enc.eval()
    with torch.no_grad():
        ref = enc(x).tokens
    exp = enc.export()
    with torch.no_grad():
        out = exp(x).tokens
    err = float((out - ref).abs().max() / ref.abs().max())
    t_eval = bench(enc, x, a.reps)
    t_exp = bench(exp, x, a.reps)
    res = {
        "encoder": {"type": a.type, "N": a.N, "fields": a.fields},
        "size": a.size,
        "max_rel_err_tokens": err,
        "pass": err < 1e-4,
        "img_per_s_escnn_train_mode": t_train_mode,
        "img_per_s_escnn_eval": t_eval,
        "img_per_s_exported": t_exp,
        "speedup_vs_eval": t_exp / t_eval,
        "speedup_vs_train_mode": t_exp / t_train_mode,
        "device": "cpu (4-core VM), fp32",
        "date": time.strftime("%Y-%m-%d"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = "export.json" if a.type == "steerable" else f"export_{a.type}.json"
    (OUT / name).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
