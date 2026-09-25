"""Overfit test: N training samples must reach >= 99% exact match on themselves within 3000
steps (Phase 2 / Phase 3 DoD). Writes benchmarks/train/overfit_<name>.json.

    python scripts/overfit.py --name baseline --set data.image_size=128
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from ouroboros.eval.metrics import score_pair
from ouroboros.train.config import load_config
from ouroboros.train.trainer import Trainer

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "train"


def exact_on(trainer: Trainer, n: int, bs: int = 64) -> float:
    trainer.model.eval()
    ds = trainer.synth
    hits = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            items = [ds[j] for j in range(i, min(n, i + bs))]
            imgs = torch.stack([x["image"] for x in items]).to(trainer.dev)
            with torch.autocast(trainer.dev.type, torch.bfloat16, enabled=False):
                preds = trainer.model.generate(imgs)
            hits += sum(score_pair(p, x["smiles"]).exact for p, x in zip(preds, items, strict=True))
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    overrides = [
        "data.synthetic_root=data/full/shards",
        f"data.synthetic_max_samples={a.n}",
        "data.num_workers=0",
        "train.batch_size=32",
        f"train.steps={a.max_steps}",
        "train.lr=5.0e-4",
        "train.warmup=100",
        "train.amp=none",
        "train.log_every=50",
        f"train.ckpt_every={10 * a.max_steps}",
        *a.set,
    ]
    cfg = load_config(a.config, overrides)
    history = []
    t0 = time.time()

    class Done(Exception):
        pass

    def cb(tr: Trainer, loss: float):
        if tr.step % a.eval_every == 0:
            em = exact_on(tr, a.n)
            history.append({"step": tr.step, "loss": loss, "exact": em, "sec": time.time() - t0})
            print(history[-1], flush=True)
            if em >= 0.99:
                raise Done

    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(cfg, tmp)
        try:
            tr.fit(callback=cb)
        except Done:
            pass
    best = max(h["exact"] for h in history) if history else 0.0
    res = {
        "name": a.name,
        "n_samples": a.n,
        "overrides": overrides,
        "encoder": cfg["model"]["encoder"],
        "history": history,
        "steps_to_99": next((h["step"] for h in history if h["exact"] >= 0.99), None),
        "best_exact": best,
        "pass": best >= 0.99 and history[-1]["step"] <= 3000,
        "device": str(tr.dev),
        "date": time.strftime("%Y-%m-%d"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"overfit_{a.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
