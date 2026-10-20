"""Build leakage-controlled real-document splits from the VERDICT-PMC snapshot.

    python scripts/hf_to_manifest.py inspect --repo yaniguan/ocsr-dataset --local HF/ocsr  # fetch
    python scripts/make_real_splits.py --local HF/ocsr --out HF/real_v1
    python scripts/ingest_real.py --manifest HF/real_v1/manifest.csv --out HF/shards_v1

Split policy (all decisions are deterministic functions of the data; nothing depends on a model):
  * test  = the published validation split (article-disjoint from train), unchanged in size.
  * dev   = ~5% of the published train *articles* (md5 of PMCID); only for checkpoint selection.
  * train = the rest of the published train split, minus every row whose InChIKey connectivity
    block (first 14 characters) also occurs in dev or test. Evaluation sets are therefore never
    shrunk by molecule overlap; the training set pays for it.
  * dev rows whose connectivity block occurs in test are dropped from dev.

Also written: ``eval_keys.txt`` (connectivity blocks of dev+test, for ``data.exclude_keys`` so the
synthetic corpus cannot contain real evaluation molecules) and ``split_stats.json``.
Output must stay out of git: images are CC BY / CC BY-NC (DATA.md).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

DEV_ARTICLE_PERCENT = 5


def _bucket(pmcid: str) -> int:
    return int(hashlib.md5(pmcid.encode()).hexdigest(), 16) % 100


def _rows(local: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for f in sorted(glob.glob(str(local / f"data/{split}-*.parquet"))):
        table = pq.read_table(f).to_pylist()
        for i, r in enumerate(table):
            r["_file"], r["_row"] = Path(f).name, i
            rows.append(r)
    return rows


def build(local: Path, out: Path) -> dict:
    train, test = _rows(local, "train"), _rows(local, "validation")
    dev = [r for r in train if _bucket(r["pmcid"]) < DEV_ARTICLE_PERCENT]
    train = [r for r in train if _bucket(r["pmcid"]) >= DEV_ARTICLE_PERCENT]
    key = lambda r: r["inchikey"][:14]  # noqa: E731
    test_keys = {key(r) for r in test}
    n_dev0 = len(dev)
    dev = [r for r in dev if key(r) not in test_keys]
    eval_keys = test_keys | {key(r) for r in dev}
    n_train0 = len(train)
    train = [r for r in train if key(r) not in eval_keys]
    assert not ({key(r) for r in train} & eval_keys)
    assert not ({r["pmcid"] for r in train} & {r["pmcid"] for r in test})

    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, rows in (("train", train), ("val", dev), ("test", test)):
        for r in rows:
            rel = Path("images") / name / f"{r['pmcid']}_{r['figure_id']}_{r['crop_index']}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(r["image"]["bytes"])
            manifest.append(
                {
                    "image": str(rel),
                    "smiles": r["smiles"],
                    "source": "verdict-pmc",
                    "split": name,
                    "id": f"{r['_file']}#{r['_row']}",
                    "pmcid": r["pmcid"],
                    "n_agree": r["n_agree"],
                    "n_agree_open": r["n_agree_open"],
                    "inchikey": r["inchikey"],
                }
            )
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    (out / "eval_keys.txt").write_text("\n".join(sorted(eval_keys)) + "\n")
    stats = {
        "policy": "test=published validation; dev=5% train articles by md5(pmcid); "
        "train minus connectivity blocks of dev+test",
        "rows": {"train": len(train), "val": len(dev), "test": len(test)},
        "articles": {
            n: len({r["pmcid"] for r in rows})
            for n, rows in (("train", train), ("val", dev), ("test", test))
        },
        "removed_from_train_for_overlap": n_train0 - len(train),
        "removed_from_dev_for_overlap": n_dev0 - len(dev),
        "n_agree": {
            n: dict(sorted(Counter(r["n_agree"] for r in rows).items()))
            for n, rows in (("train", train), ("val", dev), ("test", test))
        },
        "eval_connectivity_blocks": len(eval_keys),
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.local, a.out), indent=1))


if __name__ == "__main__":
    main()
