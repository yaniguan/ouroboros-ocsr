"""Evaluation: predictions on rendered and real-document sets, with a rotation sweep.

Output of ``run_eval``: a per-sample JSONL (``set``, ``kind`` = "rendered" | "real", ``key``,
``angle``, ``ref``, ``pred`` and the per-pair metrics) and a summary JSON with one row per
(set, angle): sample count, metric means and a bootstrap 95% CI for exact match.

Reporting rules (Am1-D): rendered and real sets are always separate rows/columns; nothing in this
module averages across kinds, and ``ouroboros.eval.aggregate`` refuses to. Every summary carries
the scoring-parity footnote until parity with arXiv:2608.09100 is verified.

Rotation convention: counter-clockwise about the image centre (``rotate_images``); multiples of
90 deg are exact pixel permutations, other angles are bilinear. An angle is "on-grid" for C_N if it
is a multiple of 360/N. Only multiples of 90 deg are exact symmetries of the pixel grid; for C8/C16
the extra on-grid angles still involve interpolation (reported, not assumed exact).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from ouroboros.data.loader import rotate_images
from ouroboros.eval.metrics import aggregate, bootstrap_ci, score_pair
from ouroboros.eval.standardize import PARITY_FOOTNOTE, PARITY_VERIFIED

KINDS = ("rendered", "real")
SWEEP_ANGLES = tuple(range(0, 360, 15))


def on_grid(angle: float, n: int) -> bool:
    step = 360.0 / n
    r = angle % step
    return min(r, step - r) < 1e-6


def angle_flags(angle: float) -> dict:
    return {
        "pixel_exact": on_grid(angle, 4),
        "on_grid_C4": on_grid(angle, 4),
        "on_grid_C8": on_grid(angle, 8),
        "on_grid_C16": on_grid(angle, 16),
    }


def predict(
    generate: Callable[[torch.Tensor], list[str]],
    dataset,
    angles: Sequence[float] = (0.0,),
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    max_samples: int | None = None,
):
    """Yield (index, angle, pred) for every sample and angle."""
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for start in range(0, n, batch_size):
        idx = list(range(start, min(n, start + batch_size)))
        imgs = torch.stack([dataset[i]["image"] for i in idx]).to(device)
        for a in angles:
            rot = rotate_images(imgs, torch.full((len(idx),), float(a))) if a % 360 else imgs
            for i, p in zip(idx, generate(rot), strict=True):
                yield i, float(a), p


def run_eval(
    generate: Callable[[torch.Tensor], list[str]],
    sets: dict[str, tuple[object, str]],
    out_dir: str | Path,
    angles: Sequence[float] = (0.0,),
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    max_samples: int | None = None,
    n_boot: int = 1000,
) -> dict:
    """``sets`` maps name -> (dataset, kind). Writes ``predictions.jsonl`` and ``summary.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "predictions.jsonl", "w") as f:
        for name, (ds, kind) in sets.items():
            if kind not in KINDS:
                raise ValueError(f"set {name}: kind must be one of {KINDS}")
            for i, a, pred in predict(generate, ds, angles, batch_size, device, max_samples):
                meta = ds.meta(i)
                sc = score_pair(pred, meta["smiles"])
                rec = {"set": name, "kind": kind, "index": i, "angle": a, "ref": meta["smiles"]}
                rec.update(pred=pred, **asdict(sc))
                rows.append(rec)
                f.write(json.dumps(rec) + "\n")
    summary = summarize(rows, n_boot=n_boot)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def summarize(rows: list[dict], n_boot: int = 1000) -> dict:
    """One entry per (set, angle); never pools different sets or kinds."""
    from ouroboros.eval.metrics import PairScore

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["set"], r["kind"], r["angle"]), []).append(r)
    fields = PairScore.__dataclass_fields__
    entries = []
    for (name, kind, angle), rs in sorted(groups.items()):
        scores = [PairScore(**{k: r[k] for k in fields}) for r in rs]
        agg = aggregate(scores)
        mean, lo, hi = bootstrap_ci([s.exact for s in scores], n_resamples=n_boot)
        entries.append(
            {
                "set": name,
                "kind": kind,
                "angle": angle,
                **angle_flags(angle),
                **agg,
                "exact_ci95": [lo, hi],
                "n_boot": n_boot,
            }
        )
    return {
        "entries": entries,
        "parity_verified": PARITY_VERIFIED,
        "footnote": None if PARITY_VERIFIED else PARITY_FOOTNOTE,
    }
