"""Run released checkpoint(s) on ingested real-document shards -> predictions CSV for the harness.

    python scripts/predict_real.py --ckpt checkpoints/ouroboros-C.pt --shards real/subset100 \
        --out runs/real_subset100/C_preds.csv
    python scripts/eval_predictions.py runs/real_subset100/C_preds.csv \
        --json runs/real_subset100/C_report.json --md runs/real_subset100/C_report.md

Output columns: ``id,gold,pred,subset`` (``subset`` = manifest ``source``). Angle 0, CPU unless
CUDA is available. Checkpoints contain RNG state: only load trusted files.

  --unlabeled        shards may have no ``smiles`` metadata (images only); ``gold`` is written
                     as an empty string. Score such a file only with
                     ``eval_predictions.py --skip-unlabeled`` (it refuses empty gold otherwise).

Decoding (defaults = one checkpoint, greedy, no TTA: the original behaviour):
  --beam K           beam search, K hypotheses per image and source
  --rerank valid     take the best RDKit-valid candidate (default when K > 1); ``none`` = best score
  --ckpt A --ckpt B  ensemble: every model proposes candidates on every view
  --tta pad,sharpen  extra input views: ``pad`` shrinks the drawing to 90% and recentres it on
                     white, ``sharpen`` applies an unsharp mask
  --nbest-out F      per-image ranked candidates as JSON lines
With several sources (models x views), every unique candidate is rescored by every source and
ranked by its mean length-normalized log probability; spellings of one molecule are merged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from ouroboros.data.loader import ShardDataset
from ouroboros.decode.ensemble import TTA, predict_batch, tta_pad, tta_sharpen  # noqa: F401
from ouroboros.decode.rerank import Candidate
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


def _nbest_record(meta: dict, i: int, gold: str, pred: str, ranked: list[Candidate]) -> dict:
    from ouroboros.eval.smiles import is_valid

    return {
        "id": meta.get("id", i),
        "gold": gold,
        "subset": meta.get("source", ""),
        "pred": pred,
        "candidates": [
            {"smiles": s, "score": score if math.isfinite(score) else None, "valid": is_valid(s)}
            for s, score in ranked
        ],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint; repeat for an ensemble")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--unlabeled", action="store_true",
                    help="shards without smiles metadata; write gold as empty")
    ap.add_argument("--beam", type=int, default=1, help="beam size (1 = greedy)")
    ap.add_argument("--rerank", choices=["valid", "none"], default=None,
                    help="candidate selection (default: valid if --beam > 1, else none)")
    ap.add_argument("--tta", default="", help=f"comma list of extra views among {sorted(TTA)}")
    ap.add_argument("--nbest-out", default=None, help="JSON lines of ranked candidates per image")
    ap.add_argument("--min-margin", type=float, default=None)
    ap.add_argument("--max-disagreement", type=float, default=None)
    a = ap.parse_args(argv)
    from ouroboros.decode.confidence import confidence
    gated = a.min_margin is not None or a.max_disagreement is not None
    confidence('', [], min_margin=a.min_margin, max_disagreement=a.max_disagreement)
    if a.beam < 1:
        ap.error("--beam must be >= 1")
    tta = [t for t in a.tta.split(",") if t]
    if unknown := sorted(set(tta) - set(TTA)):
        ap.error(f"unknown --tta views {unknown}; choose among {sorted(TTA)}")
    prefer_valid = (a.rerank or ("valid" if a.beam > 1 else "none")) == "valid"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_checkpoint(p, dev) for p in a.ckpt]
    models, cfg = [m for m, _ in loaded], loaded[0][1]
    size = cfg["data"]["image_size"]
    if any(c["data"]["image_size"] != size for _, c in loaded):
        raise SystemExit("ensemble checkpoints must share data.image_size")
    greedy = a.beam == 1 and len(models) == 1 and not tta and a.nbest_out is None and not gated
    ds = ShardDataset(a.shards, a.split, image_size=size, max_samples=a.max_samples,
                      labeled=not a.unlabeled)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    nbest_f = None
    if a.nbest_out:
        Path(a.nbest_out).parent.mkdir(parents=True, exist_ok=True)
        nbest_f = open(a.nbest_out, "w")  # noqa: SIM115
    t0 = time.time()
    try:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "subset"] +
                               (["accepted", "rejection_reason", "margin", "disagreement"] if gated else []))
            w.writeheader()
            for start in range(0, len(ds), a.batch_size):
                idx = range(start, min(len(ds), start + a.batch_size))
                imgs = torch.stack([ds[i]["image"] for i in idx]).to(dev)
                source_predictions = []
                with torch.inference_mode():
                    if greedy:
                        results = [(p, None) for p in models[0].generate(imgs)]
                    else:
                        results = predict_batch(models, imgs, a.beam, tta, prefer_valid,
                                                diagnostics=source_predictions)
                for i, (p, ranked) in zip(idx, results, strict=True):
                    m = ds.meta(i)
                    gold = "" if a.unlabeled else m["smiles"]
                    gate = confidence(p, ranked or [], source_predictions[i-start] if source_predictions else [],
                                      min_margin=a.min_margin, max_disagreement=a.max_disagreement)
                    w.writerow({"id": m.get("id", i), "gold": gold, "pred": p,
                                "subset": m.get("source", "")} | (gate if gated else {}))
                    if nbest_f is not None:
                        nbest_f.write(json.dumps(_nbest_record(m, i, gold, p, ranked) |
                            {"subset": m.get("source", ""), "source_predictions": source_predictions[i-start],
                             "confidence": gate}) + "\n")
                print(f"{min(len(ds), start + a.batch_size)}/{len(ds)}  {time.time() - t0:.0f}s",
                      flush=True)
    finally:
        if nbest_f is not None:
            nbest_f.close()
    print(f"wrote {len(ds)} predictions to {a.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
