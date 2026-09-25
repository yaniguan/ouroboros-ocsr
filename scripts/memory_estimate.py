"""Training-memory estimate for batch B at 384 px in bf16 (Phase 5 DoD: B >= 32 fits in 40 GB).

Two parts:
1. Analytic: static memory = 16 bytes/param (fp32 weights + fp32 grads + Adam m, v).
2. Activations: measured on CPU by summing the bytes of every tensor autograd saves for backward
   (``torch.autograd.graph.saved_tensors_hooks``, unique storages) during one forward of the
   full model (encoder + shared decoder + loss) at batch 1 and 2, fp32. Linear in batch size, so
   A(B) = A1 + (B - 1) * (A2 - A1). bf16 autocast stores most activations in 2 bytes; we report the
   fp32 number as an upper bound and fp32/2 as the bf16 estimate. The CPU math attention kernel
   saves the [B, H, T, T] attention matrices (flash / memory-efficient CUDA kernels do not), which
   also makes the estimate conservative. A 20% allocator/workspace margin is added.

    python scripts/memory_estimate.py --encoder equiv_attention --batch 32
Writes benchmarks/encoders/memory_<encoder>.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import build_model

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "encoders"
ENCODERS = {
    "baseline": {"type": "baseline"},
    "steerable": {"type": "steerable", "N": 8, "fields": [7, 14, 27, 55]},
    "equiv_attention": {
        "type": "equiv_attention",
        "N": 8,
        "fields": [7, 14, 27, 55],
        "attn_dim": 160,
        "attn_ff": 640,
    },
}


def saved_bytes(model, images, ids) -> int:
    seen: dict[int, int] = {}

    def pack(t):
        try:
            st = t.untyped_storage()
            seen[st.data_ptr()] = max(seen.get(st.data_ptr(), 0), st.nbytes())
        except Exception:  # noqa: BLE001 - e.g. tensors without storage
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model.loss(images, ids)
    loss.backward()
    return sum(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="equiv_attention", choices=sorted(ENCODERS))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=75, help="label length incl. BOS/EOS (max 73)")
    ap.add_argument("--gpu-gb", type=float, default=40.0)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = SmilesTokenizer.load("configs/vocab.json")
    cfg = {"model": {"encoder": ENCODERS[a.encoder], "decoder": {}}, "data": {"image_size": a.size}}
    model = build_model(cfg, tok).train()
    n_params = sum(p.numel() for p in model.parameters())
    meas = {}
    for b in (1, 2):
        x = torch.rand(b, 1, a.size, a.size)
        ids = torch.randint(4, len(tok), (b, a.seq_len))
        t0 = time.time()
        meas[b] = saved_bytes(model, x, ids)
        model.zero_grad(set_to_none=True)
        print(f"batch {b}: saved activations {meas[b] / 2**30:.3f} GiB ({time.time() - t0:.0f}s)")
    per_sample = meas[2] - meas[1]
    act_fp32 = meas[1] + (a.batch - 1) * per_sample
    static = 16 * n_params
    gib = 2**30
    est_bf16 = 1.2 * (static + act_fp32 / 2)
    est_fp32 = 1.2 * (static + act_fp32)
    res = {
        "encoder": ENCODERS[a.encoder],
        "image_size": a.size,
        "batch": a.batch,
        "seq_len": a.seq_len,
        "params": n_params,
        "static_GiB": static / gib,
        "activations_batch1_fp32_GiB": meas[1] / gib,
        "activations_per_sample_fp32_GiB": per_sample / gib,
        "activations_fp32_GiB": act_fp32 / gib,
        "estimate_bf16_GiB": est_bf16 / gib,
        "upper_bound_fp32_GiB": est_fp32 / gib,
        "fits_40GB_bf16": est_bf16 / 1e9 < a.gpu_gb,
        "max_batch_bf16_40GB": int(
            ((a.gpu_gb * 1e9 / 1.2 - static) * 2 - meas[1]) // per_sample + 1
        ),
        "peak_rss_GiB_local": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "method": "saved-tensor bytes at batch 1 and 2 on CPU (fp32), linear extrapolation; "
        "bf16 = fp32/2; +20% margin; 16 B/param static",
        "date": time.strftime("%Y-%m-%d"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"memory_{a.encoder}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
