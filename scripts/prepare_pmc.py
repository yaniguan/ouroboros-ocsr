"""VERDICT-PMC parquet (HF ``yaniguan/ocsr-dataset``) -> leakage-filtered real manifest.

    python scripts/prepare_pmc.py --parquet /content/hf_pmc/data --out real_data/pmc \
        --exclude-keys /content/eval_keys.txt
    python scripts/ingest_real.py --manifest real_data/pmc/manifest.csv --out real/pmc

Input: the dataset's parquet files (``train-*.parquet`` -> split ``train``,
``validation-*.parquet`` -> split ``val``; HF image struct + ``smiles`` column).

Every row is standardized exactly like ``scripts/ingest_real.py`` / the scoring harness
(``ouroboros.eval.standardize``) and then dropped, with a counted reason, when

* ``eval_overlap``: its leakage key's connectivity block (first 14 chars, i.e. stereoisomers
  included, as in ``ShardDataset(exclude_key14=...)``) is in ``--exclude-keys`` (the held-out
  real test molecules from ``scripts/check_leakage.py --dump-eval-keys``) -- applied to BOTH
  splits, so neither training nor checkpoint selection sees a test molecule;
* ``label_unparsable`` / ``label_untokenizable`` / ``label_too_long`` (``len(encode)-1 >
  --max-len``, the decoder limit);
* ``below_min_agree_open`` (optional stricter weak-label filter);
* ``val_in_train`` (``--val-disjoint``, default on): a val molecule (connectivity block) also
  present in the kept train rows, so real validation measures generalization to new molecules,
  like the held-out test set.

Val rows are written in a seeded random order so a validation *prefix* (the Trainer takes the
first ``validation.samples``) is a random, article-mixed subset. Outputs (never committed,
DATA.md): ``images/<split>/*.png``, ``manifest.csv``, ``prepare_stats.json``,
``train_tokens.json`` (token counts of kept train labels, for the append-only vocabulary).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter
from pathlib import Path

from rdkit import RDLogger

from ouroboros.decode.tokenizer import tokenize
from ouroboros.eval.standardize import leakage_key, standardize

RDLogger.DisableLog("rdApp.*")


def split_of(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("train"):
        return "train"
    if name.startswith(("validation", "valid", "val")):
        return "val"
    return None


def find_parquets(specs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in specs:
        p = Path(s)
        out += sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    return out


def iter_rows(path: Path, columns: list[str], batch_size: int = 256):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[c for c in columns if c in names]):
        yield from batch.to_pylist()


def prepare(
    parquets: list[Path],
    out: Path,
    exclude_key14: set[str],
    max_len: int = 128,
    min_agree_open: int = 0,
    val_disjoint: bool = True,
    limit: dict[str, int] | None = None,
    seed: int = 0,
) -> dict:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    limit = limit or {}
    kept: dict[str, list[dict]] = {"train": [], "val": []}
    reasons: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    n_rows: Counter = Counter()
    tokens: Counter = Counter()
    cols = ["image", "smiles", "inchikey", "n_agree", "n_agree_open", "pmcid", "figure_id",
            "crop_index"]
    # train first so --val-disjoint knows every kept train molecule
    order = sorted(parquets, key=lambda p: (split_of(p) != "train", str(p)))
    for path in order:
        split = split_of(path)
        if split is None:
            print(f"skip {path}: unknown split")
            continue
        d = out / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(iter_rows(path, cols)):
            if limit.get(split) is not None and n_rows[split] >= limit[split]:
                break
            n_rows[split] += 1
            r = reasons[split]
            if min_agree_open and (rec.get("n_agree_open") or 0) < min_agree_open:
                r["below_min_agree_open"] += 1
                continue
            std = standardize(rec.get("smiles"))
            if not std.valid:
                r["label_unparsable"] += 1
                continue
            key = leakage_key(std)
            if key is None:
                r["label_no_key"] += 1
                continue
            if key[:14] in exclude_key14:
                r["eval_overlap"] += 1
                continue
            try:
                toks = tokenize(std.smiles)
            except ValueError:
                r["label_untokenizable"] += 1
                continue
            if len(toks) + 1 > max_len:  # == len(encode(label)) - 1 > max_len
                r["label_too_long"] += 1
                continue
            img = rec.get("image") or {}
            data = img.get("bytes") if isinstance(img, dict) else img
            if not data:
                r["image_missing"] += 1
                continue
            rid = f"{rec.get('pmcid')}_{rec.get('figure_id')}_{rec.get('crop_index')}_{path.stem}_{i}"
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in rid)
            p = d / f"{safe}.png"
            if not p.exists():
                try:
                    Image.open(io.BytesIO(data)).convert("RGB").save(p)
                except Exception:  # noqa: BLE001 - PIL raises assorted errors
                    r["image_unreadable"] += 1
                    continue
            kept[split].append(
                {"image": str(p.relative_to(out)), "smiles": rec["smiles"], "source": "PMC",
                 "split": split, "id": rid, "_key14": key[:14]}
            )
            if split == "train":
                tokens.update(toks)
    if val_disjoint:
        train14 = {r["_key14"] for r in kept["train"]}
        before = len(kept["val"])
        kept["val"] = [r for r in kept["val"] if r["_key14"] not in train14]
        reasons["val"]["val_in_train"] += before - len(kept["val"])
    random.Random(seed).shuffle(kept["val"])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "smiles", "source", "split", "id"])
        w.writeheader()
        for split in ("train", "val"):
            for r in kept[split]:
                w.writerow({k: v for k, v in r.items() if not k.startswith("_")})
    stats = {
        "parquets": [str(p) for p in order],
        "n_exclude_key14": len(exclude_key14),
        "max_len": max_len,
        "min_agree_open": min_agree_open,
        "val_disjoint": val_disjoint,
        "rows_read": dict(n_rows),
        "kept": {s: len(v) for s, v in kept.items()},
        "kept_unique_key14": {s: len({r["_key14"] for r in v}) for s, v in kept.items()},
        "dropped": {s: dict(c) for s, c in reasons.items()},
    }
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "train_tokens.json").write_text(json.dumps(dict(tokens.most_common()), indent=0))
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parquet", nargs="+", required=True, help="parquet files or directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-keys", required=True,
                    help="held-out eval InChIKeys (check_leakage.py --dump-eval-keys)")
    ap.add_argument("--max-len", type=int, default=128, help="decoder max_len")
    ap.add_argument("--min-agree-open", type=int, default=0)
    ap.add_argument("--no-val-disjoint", action="store_true")
    ap.add_argument("--limit-train", type=int, default=None, help="read only the first N rows")
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    keys = Path(a.exclude_keys).read_text().split()
    if not keys:
        raise SystemExit(f"{a.exclude_keys} has no keys; refusing to run without test exclusion")
    stats = prepare(
        find_parquets(a.parquet),
        Path(a.out),
        {k[:14] for k in keys},
        max_len=a.max_len,
        min_agree_open=a.min_agree_open,
        val_disjoint=not a.no_val_disjoint,
        limit={"train": a.limit_train, "val": a.limit_val},
        seed=a.seed,
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
