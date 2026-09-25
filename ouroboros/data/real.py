"""Real-document data adapter: manifest of (image, SMILES) -> the synthetic shard format.

Manifest interface
------------------
A CSV (``.csv``) or JSON-lines (``.jsonl``) file with one row per image:

    column   required  meaning
    -------  --------  -----------------------------------------------------------------
    image    yes       path to the image (PNG/JPEG/TIFF/...); relative paths are resolved
                       against the manifest's directory
    smiles   yes       reference SMILES (any valid RDKit SMILES; standardized on ingest)
    source   no        dataset name, e.g. "USPTO", "ACS", "CLEF-IP" (default: manifest stem)
    split    no        "train" | "val" | "test" (default: "test")
    id       no        original identifier, kept in the sample metadata

Output (per ``out_dir``): ``{split}-{i:06d}.tar`` WebDataset shards with ``{key}.png`` and
``{key}.json`` exactly like synthetic shards, so ``ShardDataset(out_dir, split)`` reads them.
JSON metadata: ``smiles`` (standardized label), ``raw_smiles``, ``source``, ``id``, ``real: 1``,
``orig_size``.

Labels are standardized with ``ouroboros.eval.standardize.standardize`` (the scoring
standardization, Am1-C). Rows whose label or image cannot be processed are NOT silently dropped:
each is written to ``ingest_failures.jsonl`` with its reason and counted in ``ingest_stats.json``.

Image geometry: the original image is scaled (aspect preserved) so that its whole rectangle fits
inside the inscribed circle of the ``size x size`` canvas and is centred on a white background
-- the same convention as synthetic renderings, so rotation never crops content.
"""

from __future__ import annotations

import csv
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image

from ouroboros.data.build import _add
from ouroboros.data.render import encode_png
from ouroboros.eval.standardize import standardize

SPLITS = ("train", "val", "test")


def read_real_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        missing = {"image", "smiles"} - set(r)
        if missing:
            raise ValueError(f"{path}: row {i} lacks required column(s) {sorted(missing)}")
        r.setdefault("source", path.stem)
        r["source"] = r["source"] or path.stem
        r["split"] = (r.get("split") or "test").strip()
        if r["split"] not in SPLITS:
            raise ValueError(f"{path}: row {i} has split {r['split']!r}, expected one of {SPLITS}")
        img = Path(r["image"])
        r["image_path"] = str(img if img.is_absolute() else (path.parent / img))
        r["row"] = i
    return rows


def fit_to_canvas(img: Image.Image, size: int = 384) -> Image.Image:
    """Grayscale; scale so the image diagonal equals the canvas side (fits the inscribed
    circle); centre on white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("L")
    w, h = img.size
    scale = size / math.hypot(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS if scale < 1 else Image.BICUBIC)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def ingest(
    manifest: str | Path,
    out_dir: str | Path,
    size: int = 384,
    shard_size: int = 1000,
    key_prefix: str | None = None,
) -> dict:
    """Convert a real-data manifest into shards. Returns (and writes) ingest statistics."""
    rows = read_real_manifest(manifest)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = key_prefix or f"real-{Path(manifest).stem}"
    failures: list[dict] = []
    reasons: Counter = Counter()
    written: Counter = Counter()
    for split in SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        tar = None
        n_in_shard = shard_idx = 0
        for idx, r in enumerate(split_rows):
            std = standardize(r["smiles"])
            reason = None if std.valid else f"label_{std.reason}"
            img = None
            if reason is None:
                try:
                    with Image.open(r["image_path"]) as im:
                        orig_size = im.size
                        img = fit_to_canvas(im, size)
                except (OSError, ValueError) as e:
                    reason = f"image_unreadable:{type(e).__name__}"
            if reason is not None:
                reasons[reason] += 1
                failures.append(
                    {
                        "row": r["row"],
                        "id": r.get("id"),
                        "image": r["image"],
                        "raw_smiles": r["smiles"],
                        "split": split,
                        "reason": reason,
                    }
                )
                continue
            if tar is None or n_in_shard >= shard_size:
                if tar is not None:
                    tar.close()
                tar = tarfile.open(
                    out_dir / f"{split}-{shard_idx:06d}.tar", "w", format=tarfile.USTAR_FORMAT
                )
                shard_idx += 1
                n_in_shard = 0
            key = f"{prefix}_{split}_{idx:08d}".replace(".", "-")
            meta = {
                "smiles": std.smiles,
                "raw_smiles": r["smiles"],
                "source": r["source"],
                "id": r.get("id"),
                "real": 1,
                "orig_size": list(orig_size),
            }
            _add(tar, f"{key}.png", encode_png(img))
            _add(tar, f"{key}.json", json.dumps(meta).encode())
            n_in_shard += 1
            written[split] += 1
        if tar is not None:
            tar.close()
    with open(out_dir / "ingest_failures.jsonl", "w") as f:
        for fl in failures:
            f.write(json.dumps(fl) + "\n")
    stats = {
        "manifest": str(manifest),
        "n_rows": len(rows),
        "written": dict(written),
        "n_failures": len(failures),
        "failure_reasons": dict(reasons),
        "label_parse_rate": 1.0
        - sum(v for k, v in reasons.items() if k.startswith("label_")) / max(1, len(rows)),
    }
    (out_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=1))
    return stats
