"""Equivariance errors of the steerable encoders at full resolution (Phase 3 / 5 DoD).

For each N: random-init encoder with non-trivial BN statistics, eval mode, fp32.
* 90-deg multiples (exact on the pixel grid): max relative error of trunk feature maps
  (escnn action on the output vs. features of the rotated input) and of the token set after
  undoing the known grid permutation.
* Off-grid angles (multiples of 360/N that are not multiples of 90; report only): escnn's
  interpolated action on input and output feature maps (central disk only, to exclude the
  corners that leave the frame), and the relative change of the permutation-invariant mean token.

    python scripts/measure_equivariance.py --size 384
Writes benchmarks/encoders/equivariance.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from escnn import nn as enn

from ouroboros.encoder.steerable import SteerableConfig, SteerableEncoder

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "encoders"


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def disk(h, w, frac=0.7):
    ys, xs = torch.meshgrid(
        torch.arange(h) - (h - 1) / 2, torch.arange(w) - (w - 1) / 2, indexing="ij"
    )
    return (xs**2 + ys**2) <= (frac * min(h, w) / 2) ** 2


def build(kind: str, N: int, size: int, fields):
    torch.manual_seed(0)
    if kind == "steerable":
        enc = SteerableEncoder(SteerableConfig(N=N, fields=fields, image_size=size))
    else:
        from ouroboros.encoder.equiv_attention import EquivAttentionEncoder, EquivAttnConfig

        enc = EquivAttentionEncoder(EquivAttnConfig(N=N, fields=fields, image_size=size))
    enc.train()
    with torch.no_grad():
        for _ in range(2):
            enc(torch.rand(2, 1, size, size))
    return enc.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--N", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--kind", default="steerable", choices=["steerable", "equiv_attention"])
    ap.add_argument("--channels", type=int, nargs=4, default=[56, 112, 232, 464])
    a = ap.parse_args()
    torch.manual_seed(1)
    # a smooth test image (random blobs) so that off-grid interpolation error is meaningful
    x = torch.nn.functional.avg_pool2d(torch.rand(1, 1, a.size, a.size), 9, 1, 4)
    x = x * disk(a.size, a.size, 0.95).float()
    res = {"size": a.size, "kind": a.kind, "results": {}}
    for N in a.N:
        fields = [max(1, c // N) for c in a.channels]
        enc = build(a.kind, N, a.size, fields)
        g = enc.gspace.fibergroup
        out = {"fields": fields, "grid90": {}, "offgrid": {}}
        with torch.no_grad():
            f = enc.features(x)
            t = enc(x).tokens
            h = t.shape[1]
            side = int(round(h**0.5))
            idx = torch.arange(h).view(1, side, side)
            for k in (1, 2, 3):
                el = g.element(k * N // 4)
                xr = torch.rot90(x, k, dims=(-2, -1))
                fr = enc.features(xr)
                tr = enc(xr).tokens
                perm = torch.rot90(idx, k, dims=(-2, -1)).reshape(-1)
                out["grid90"][f"{90 * k}"] = {
                    "feature_rel_err": rel(f.transform(el).tensor, fr.tensor),
                    "token_rel_err": rel(tr, t[:, perm]),
                }
            for j in range(1, N):
                if (4 * j) % N == 0:
                    continue
                el = g.element(j)
                xin = enn.GeometricTensor(x, enc.in_type).transform(el)
                fr = enc.features(xin.tensor)
                ft = f.transform(el).tensor
                m = disk(*ft.shape[-2:], 0.7)
                tr = enc(xin.tensor).tokens
                out["offgrid"][f"{360 * j / N:g}"] = {
                    "feature_rel_err_central_disk": rel(ft[..., m], fr.tensor[..., m]),
                    "mean_token_rel_err": rel(tr.mean(1), t.mean(1)),
                }
        res["results"][f"C{N}"] = out
        print(N, json.dumps(out), flush=True)
    res["date"] = time.strftime("%Y-%m-%d")
    OUT.mkdir(parents=True, exist_ok=True)
    name = "equivariance.json" if a.kind == "steerable" else "equivariance_attention.json"
    (OUT / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
