"""Run a released checkpoint on ingested real-document shards -> predictions CSV for the harness.

    python scripts/predict_real.py --ckpt checkpoints/ouroboros-C.pt --shards real/subset100 \
        --out runs/real_subset100/C_preds.csv
    python scripts/eval_predictions.py runs/real_subset100/C_preds.csv \
        --json runs/real_subset100/C_report.json --md runs/real_subset100/C_report.md

Output columns: ``id,gold,pred,subset`` (``subset`` = manifest ``source``). Greedy decoding,
angle 0, CPU unless CUDA is available. Checkpoints contain RNG state: only load trusted files.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import build_model, load_model_state


def load_checkpoint(path: str | Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    tok = SmilesTokenizer(state["vocab"])
    model = build_model(state["config"], tok)
    load_model_state(model, state["model"])
    model.to(device).eval()
    if hasattr(model.encoder, "export"):  # steerable encoder: pure-PyTorch inference copy
        model.encoder = model.encoder.export().to(device)
    return model, state["config"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.ckpt, dev)
    ds = ShardDataset(a.shards, a.split, image_size=cfg["data"]["image_size"],
                      max_samples=a.max_samples)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "subset"])
        w.writeheader()
        for start in range(0, len(ds), a.batch_size):
            idx = range(start, min(len(ds), start + a.batch_size))
            imgs = torch.stack([ds[i]["image"] for i in idx]).to(dev)
            with torch.inference_mode():
                preds = model.generate(imgs)
            for i, p in zip(idx, preds, strict=True):
                m = ds.meta(i)
                w.writerow({"id": m.get("id", i), "gold": m["smiles"], "pred": p,
                            "subset": m.get("source", "")})
            print(f"{min(len(ds), start + a.batch_size)}/{len(ds)}  {time.time() - t0:.0f}s",
                  flush=True)
    print(f"wrote {len(ds)} predictions to {a.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
