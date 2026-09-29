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
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from ouroboros.data.loader import rotate_images
from ouroboros.eval.metrics import aggregate, bootstrap_ci, score_pair
from ouroboros.eval.standardize import PARITY_FOOTNOTE, PARITY_VERIFIED
from ouroboros.provenance import JsonCache, atomic_json, json_sha256

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
    identity: dict | None = None,
) -> dict:
    """Evaluate sets separately. Supplying checkpoint/code ``identity`` enables resume.

    With identity, each dataset must expose identity() for its content and order.
    Without it the callable is opaque, so results are recomputed, never reused.
    """
    if batch_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("batch_size and max_samples must be positive")
    if any(kind not in KINDS for _, kind in sets.values()):
        raise ValueError(f"kind must be one of {KINDS}")
    angles = [float(a) for a in angles]
    if not angles or len(set(angles)) != len(angles):
        raise ValueError("angles must be nonempty and unique")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = None
    manifest = out / "evaluation.json"
    if identity is not None:
        settings = {
            "schema": 1,
            "model": identity,
            "sets": [
                {"name": name, "kind": kind, "data": ds.identity()}
                for name, (ds, kind) in sets.items()
            ],
            "angles": angles,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "n_boot": n_boot,
            "device": str(device),
        }
        if manifest.exists():
            if json.loads(manifest.read_text()) != settings:
                raise ValueError("Evaluation identity changed; use a new output directory")
        elif (out / "predictions.jsonl").exists():
            raise ValueError("Legacy evaluation has no identity; use a new output directory")
        else:
            atomic_json(manifest, settings)
        cache = JsonCache(out / "batches")
    elif manifest.exists():
        raise ValueError("Resumable output requires an explicit checkpoint identity")
    evaluation_digest = json_sha256(settings) if cache else None
    rows = []
    for name, (ds, kind) in sets.items():
        n = len(ds) if max_samples is None else min(max_samples, len(ds))
        for start in range(0, n, batch_size):
            idx = list(range(start, min(n, start + batch_size)))
            for angle in angles:

                def compute(idx=idx, angle=angle, ds=ds, name=name, kind=kind):
                    imgs = torch.stack([ds[i]["image"] for i in idx]).to(device)
                    if angle % 360:
                        imgs = rotate_images(imgs, torch.full((len(idx),), angle))
                    with torch.inference_mode():
                        predictions = generate(imgs)
                    batch = []
                    for i, pred in zip(idx, predictions, strict=True):
                        meta = ds.meta(i)
                        sc = score_pair(pred, meta["smiles"])
                        batch.append(
                            {
                                "set": name,
                                "kind": kind,
                                "index": i,
                                "angle": angle,
                                "ref": meta["smiles"],
                                "pred": pred,
                                **asdict(sc),
                            }
                        )
                    return batch

                job = (
                    {
                        "evaluation": evaluation_digest,
                        "set": name,
                        "indices": idx,
                        "angle": angle,
                    }
                    if cache
                    else None
                )
                rows.extend(cache.get(job, compute) if cache else compute())
    summary = summarize(rows, n_boot=n_boot)
    # The full file is replaced only when every batch has been committed.
    with tempfile.NamedTemporaryFile(mode="w", dir=out, suffix=".part", delete=False) as f:
        tmp = Path(f.name)
        try:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, out / "predictions.jsonl")
    finally:
        tmp.unlink(missing_ok=True)
    atomic_json(out / "summary.json", summary)
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
