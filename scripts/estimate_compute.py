"""GPU-hour estimate per run and for the whole grid, from configs/sweep/index.csv and the
throughput assumptions in configs/compute_assumptions.yaml (replace them with measured values).

    python scripts/estimate_compute.py [--pruned-fractions 0 0.1 0.5]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import yaml


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="configs/sweep/index.csv")
    ap.add_argument("--assumptions", default="configs/compute_assumptions.yaml")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pruned-fractions", type=float, nargs="*", default=None)
    a = ap.parse_args(argv)
    asm = yaml.safe_load(open(a.assumptions))
    rows = list(csv.DictReader(open(a.index)))
    if a.pruned_fractions is not None:
        rows = [r for r in rows if float(r["real_fraction"]) in a.pruned_fractions]
    enc_of = {
        "A": "baseline",
        "B": "baseline",
        "C": "steerable",
        "C+": "steerable",
        "D": "equiv_attention",
        "E": "baseline",  # canonicalizer adds ~1% FLOPs
    }
    eval_h = asm["eval_images"] / asm["eval_img_per_s"] / 3600
    per_group = defaultdict(lambda: [0, 0.0])
    total = 0.0
    for r in rows:
        ips = asm["train_img_per_s"][enc_of[r["arm"]]]
        h = int(r["steps"]) * a.batch / ips / 3600 + eval_h + asm["overhead_h_per_run"]
        g = per_group[(r["arm"], int(r["size"]))]
        g[0] += 1
        g[1] += h
        total += h
    print(f"{'arm':<4}{'size':>8}{'runs':>6}{'GPU-h/run':>11}{'GPU-h':>8}")
    for (arm, size), (n, h) in sorted(per_group.items()):
        print(f"{arm:<4}{size:>8}{n:>6}{h / n:>11.2f}{h:>8.1f}")
    print(f"total: {len(rows)} runs, {total:.1f} A100-hours")


if __name__ == "__main__":
    main()
