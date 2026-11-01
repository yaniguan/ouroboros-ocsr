"""Run released checkpoint(s) on ingested real-document shards -> predictions CSV for the harness.

    python scripts/predict_real.py --ckpt checkpoints/ouroboros-C.pt --shards real/subset100 \
        --out runs/real_subset100/C_preds.csv
    python scripts/eval_predictions.py runs/real_subset100/C_preds.csv \
        --json runs/real_subset100/C_report.json --md runs/real_subset100/C_report.md

Output columns: ``id,gold,pred,subset`` (``subset`` = manifest ``source``). Angle 0, CPU unless
CUDA is available. Checkpoints contain RNG state: only load trusted files.

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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from ouroboros.data.loader import ShardDataset
from ouroboros.decode.rerank import Candidate, choose, ensemble_rank
from ouroboros.decode.tokenizer import SmilesTokenizer
from ouroboros.model import OCSRModel, build_model, load_model_state


def load_checkpoint(path: str | Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    tok = SmilesTokenizer(state["vocab"])
    model = build_model(state["config"], tok)
    load_model_state(model, state["model"])
    model.to(device).eval()
    if hasattr(model.encoder, "export"):  # steerable encoder: pure-PyTorch inference copy
        model.encoder = model.encoder.export().to(device)
    return model, state["config"]


# Test-time views of an ink batch [N, 1, H, W] (ink = 1 - gray / 255, white background = 0).
def tta_pad(images: torch.Tensor, scale: float = 0.9) -> torch.Tensor:
    """Shrink the whole image to ``scale`` and recentre it on a white (zero-ink) canvas."""
    H, W = images.shape[-2:]
    h, w = max(1, round(H * scale)), max(1, round(W * scale))
    small = F.interpolate(
        images.float(), size=(h, w), mode="bilinear", align_corners=False, antialias=True
    )
    out = torch.zeros_like(images)
    top, left = (H - h) // 2, (W - w) // 2
    out[..., top : top + h, left : left + w] = small.clamp(0.0, 1.0).to(images.dtype)
    return out


def tta_sharpen(images: torch.Tensor) -> torch.Tensor:
    """PIL unsharp mask on the grayscale rendering of each (single-channel) image."""
    out = []
    for img in images.detach().float().cpu():
        gray = ((1.0 - img[0].clamp(0.0, 1.0)) * 255.0).round().to(torch.uint8).numpy()
        sharp = Image.fromarray(gray, "L").filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        out.append(1.0 - torch.from_numpy(np.array(sharp, dtype=np.float32)) / 255.0)
    return torch.stack(out)[:, None].to(images.device, images.dtype)


TTA = {"pad": tta_pad, "sharpen": tta_sharpen}


def predict_batch(
    models: list[OCSRModel],
    images: torch.Tensor,
    beam: int,
    tta: list[str],
    prefer_valid: bool,
) -> list[tuple[str, list[Candidate]]]:
    """Per image: ``(prediction, ranked pooled candidates)`` over all (model, view) sources."""
    views = [images, *(TTA[name](images) for name in tta)]
    sources = [(m, x) for m in models for x in views]
    nbest = [m.generate_nbest(x, beam_size=beam) for m, x in sources]  # [source][image]
    n = images.shape[0]
    rescored: list[dict[str, list[float]]] = [{} for _ in range(n)]
    if len(sources) > 1:  # a single source's beam scores are already its own log probs
        flat = [
            (i, s)
            for i in range(n)
            for s in dict.fromkeys(s for per_src in nbest for s, _ in per_src[i])
        ]
        idx, smiles = [i for i, _ in flat], [s for _, s in flat]
        per_source = [m.score_smiles(x, smiles, image_index=idx).tolist() for m, x in sources]
        for j, (i, s) in enumerate(flat):
            rescored[i][s] = [scores[j] for scores in per_source]
    out = []
    for i in range(n):
        ranked = ensemble_rank([per_src[i] for per_src in nbest], rescored[i])
        pred = choose(ranked)[0] if prefer_valid else ranked[0][0]
        out.append((pred, ranked))
    return out


def _nbest_record(meta: dict, i: int, pred: str, ranked: list[Candidate]) -> dict:
    from ouroboros.eval.smiles import is_valid

    return {
        "id": meta.get("id", i),
        "gold": meta["smiles"],
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
    ap.add_argument("--beam", type=int, default=1, help="beam size (1 = greedy)")
    ap.add_argument("--rerank", choices=["valid", "none"], default=None,
                    help="candidate selection (default: valid if --beam > 1, else none)")
    ap.add_argument("--tta", default="", help=f"comma list of extra views among {sorted(TTA)}")
    ap.add_argument("--nbest-out", default=None, help="JSON lines of ranked candidates per image")
    a = ap.parse_args(argv)
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
    greedy = a.beam == 1 and len(models) == 1 and not tta and a.nbest_out is None
    ds = ShardDataset(a.shards, a.split, image_size=size, max_samples=a.max_samples)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    nbest_f = None
    if a.nbest_out:
        Path(a.nbest_out).parent.mkdir(parents=True, exist_ok=True)
        nbest_f = open(a.nbest_out, "w")  # noqa: SIM115
    t0 = time.time()
    try:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "gold", "pred", "subset"])
            w.writeheader()
            for start in range(0, len(ds), a.batch_size):
                idx = range(start, min(len(ds), start + a.batch_size))
                imgs = torch.stack([ds[i]["image"] for i in idx]).to(dev)
                with torch.inference_mode():
                    if greedy:
                        results = [(p, None) for p in models[0].generate(imgs)]
                    else:
                        results = predict_batch(models, imgs, a.beam, tta, prefer_valid)
                for i, (p, ranked) in zip(idx, results, strict=True):
                    m = ds.meta(i)
                    w.writerow({"id": m.get("id", i), "gold": m["smiles"], "pred": p,
                                "subset": m.get("source", "")})
                    if nbest_f is not None:
                        nbest_f.write(json.dumps(_nbest_record(m, i, p, ranked)) + "\n")
                print(f"{min(len(ds), start + a.batch_size)}/{len(ds)}  {time.time() - t0:.0f}s",
                      flush=True)
    finally:
        if nbest_f is not None:
            nbest_f.close()
    print(f"wrote {len(ds)} predictions to {a.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
