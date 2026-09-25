"""Resume test (Phase 2 DoD) with the default architecture: a run "killed" at step k+Δ (last
checkpoint at k) and resumed must reproduce the uninterrupted run's losses over the next 100
steps (mean relative difference <= 1%). Writes benchmarks/train/resume_<name>.json.

    python scripts/resume_test.py --name baseline --set data.image_size=128
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from ouroboros.train.config import load_config
from ouroboros.train.trainer import Trainer

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "train"


def losses(out_dir: Path) -> dict[int, float]:
    recs = [json.loads(x) for x in (out_dir / "log.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in recs if "loss" in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--kill-after", type=int, default=23, help="steps past the checkpoint")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    total = a.k + a.n
    cfg = load_config(
        a.config,
        [
            "data.synthetic_root=data/full/shards",
            "data.synthetic_max_samples=2000",
            "data.num_workers=2",
            "train.batch_size=8",
            f"train.steps={total}",
            "train.warmup=20",
            "train.amp=none",
            "train.rotation_aug=true",  # exercise the augmentation RNG path as well
            "train.log_every=1",
            f"train.ckpt_every={a.k}",
            *a.set,
        ],
    )
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        full, cut = Path(d) / "full", Path(d) / "cut"
        Trainer(cfg, full).fit()
        Trainer(cfg, cut).fit(stop_at=a.k + a.kill_after)  # disconnect: ckpt at k, then lost
        tr = Trainer(cfg, cut)
        resumed_at = tr.step
        tr.fit()
        la, lb = losses(full), losses(cut)
    steps = list(range(a.k + 1, total + 1))
    rel = np.array([abs(la[s] - lb[s]) / abs(la[s]) for s in steps])
    res = {
        "name": a.name,
        "encoder": cfg["model"]["encoder"],
        "image_size": cfg["data"]["image_size"],
        "checkpoint_step": a.k,
        "killed_at": a.k + a.kill_after,
        "resumed_at": resumed_at,
        "compared_steps": [steps[0], steps[-1]],
        "mean_rel_diff": float(rel.mean()),
        "max_rel_diff": float(rel.max()),
        "pass": bool(rel.mean() <= 0.01 and resumed_at == a.k),
        "restored": [
            "model",
            "optimizer",
            "lr scheduler",
            "python/numpy/torch RNG",
            "sampler position",
            "augmentation RNG (seed, step)",
        ],
        "wall_sec": time.time() - t0,
        "date": time.strftime("%Y-%m-%d"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"resume_{a.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
