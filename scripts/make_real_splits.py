"""Build leakage-controlled real-document splits and nested label-budget pools from VERDICT-PMC.

    python scripts/hf_to_manifest.py inspect --repo yaniguan/ocsr-dataset --local HF/ocsr  # fetch
    python scripts/make_real_splits.py --local HF/ocsr --out HF/real_v2
    for m in eval pool128 pool512 pool2048; do
        python scripts/ingest_real.py --manifest HF/real_v2/manifest_$m.csv --out HF/shards_v2/$m
    done

Scope filter (a property of the *label*, never of any model): rows are removed when the standardized
label fails to parse, contains a token outside ``configs/vocab.json``, or needs more than
``--max-tokens`` decoder positions. Counts per split and reason are reported. This keeps the task to
single, vocabulary-covered molecules; it removes ~2% of rows.

Split policy (deterministic; nothing depends on a model):
  * test  = the published validation split (article-disjoint from train).
  * dev   = ~5% of the published train *articles* (md5 of PMCID); analysis / hyper-parameters only.
  * train = the rest of the published train split, minus every row whose *standardized* InChIKey
    connectivity block (first 14 characters) also occurs in dev or test.
  * dev rows whose connectivity block occurs in test are dropped from dev.
  * pools: train articles are put in a fixed pseudo-random order (md5 of PMCID); each article
    contributes at most ``--per-article`` crops (also in md5 order); the first K rows form the pool
    of K independent labels. Pools are nested (K=128 is a prefix of K=512, and so on), so a larger
    budget only adds labels, and no article can dominate a small budget.

Also written: ``eval_keys.txt`` (standardized connectivity blocks of dev+test, for
``data.exclude_keys``), ``split_stats.json``. Output must stay out of git (CC BY / CC BY-NC).
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
POOLS = (128, 512, 2048)


def _md5(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def _rows(local: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for f in sorted(glob.glob(str(local / f"data/{split}-*.parquet"))):
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            r["_file"], r["_row"] = Path(f).name, i
            rows.append(r)
    return rows


def _scope(rows: list[dict], tok, max_tokens: int) -> tuple[list[dict], Counter]:
    """Attach the standardized label / key and drop out-of-scope rows (reasons are counted)."""
    from ouroboros.decode.tokenizer import tokenize
    from ouroboros.eval.standardize import standard_inchikey, standardize

    kept, dropped = [], Counter()
    for r in rows:
        std = standardize(r["smiles"])
        if not std.valid:
            dropped["label_unparseable"] += 1
            continue
        tokens = tokenize(std.smiles)
        if any(t not in tok.stoi for t in tokens):
            dropped["out_of_vocabulary"] += 1
            continue
        if len(tokens) + 2 > max_tokens:  # BOS + EOS
            dropped["over_length"] += 1
            continue
        key = standard_inchikey(std)
        if not key:
            dropped["no_inchikey"] += 1
            continue
        r["_key14"] = key[:14]
        kept.append(r)
    return kept, dropped


def _write(out: Path, name: str, split_of: dict[str, list[dict]]) -> None:
    fields = ["image", "smiles", "source", "split", "id", "pmcid"]
    fields += ["n_agree", "n_agree_open", "inchikey"]
    with open(out / f"manifest_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split, rows in split_of.items():
            for r in rows:
                rel = Path("images") / f"{r['pmcid']}_{r['figure_id']}_{r['crop_index']}.png"
                path = out / rel
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r["image"]["bytes"])
                w.writerow(
                    {
                        "image": str(rel),
                        "smiles": r["smiles"],
                        "source": "verdict-pmc",
                        "split": split,
                        "id": f"{r['_file']}#{r['_row']}",
                        "pmcid": r["pmcid"],
                        "n_agree": r["n_agree"],
                        "n_agree_open": r["n_agree_open"],
                        "inchikey": r["inchikey"],
                    }
                )


def build(local: Path, out: Path, vocab: Path, max_tokens: int, per_article: int) -> dict:
    from ouroboros.decode.tokenizer import SmilesTokenizer

    tok = SmilesTokenizer.load(vocab)
    out.mkdir(parents=True, exist_ok=True)
    train, d_train = _scope(_rows(local, "train"), tok, max_tokens)
    test, d_test = _scope(_rows(local, "validation"), tok, max_tokens)
    in_dev = lambda r: _md5(r["pmcid"]) % 100 < DEV_ARTICLE_PERCENT  # noqa: E731
    dev = [r for r in train if in_dev(r)]
    train = [r for r in train if not in_dev(r)]
    test_keys = {r["_key14"] for r in test}
    n_dev0, dev = len(dev), [r for r in dev if r["_key14"] not in test_keys]
    eval_keys = test_keys | {r["_key14"] for r in dev}
    n_train0 = len(train)
    train = [r for r in train if r["_key14"] not in eval_keys]
    assert not ({r["_key14"] for r in train} & eval_keys)
    assert not ({r["pmcid"] for r in train} & {r["pmcid"] for r in test})

    by_article: dict[str, list[dict]] = {}
    for r in sorted(train, key=lambda r: _md5(f"{r['pmcid']}|{r['figure_id']}|{r['crop_index']}")):
        by_article.setdefault(r["pmcid"], []).append(r)
    ordered = [
        r
        for pmcid in sorted(by_article, key=lambda p: _md5("pool|" + p))
        for r in by_article[pmcid][:per_article]
    ]
    pools = {k: ordered[:k] for k in POOLS}
    assert all(len(p) == k for k, p in pools.items()), "not enough rows for the largest pool"
    for small, large in zip(POOLS, POOLS[1:], strict=False):
        assert pools[small] == pools[large][:small], "pools must be nested"

    _write(out, "eval", {"val": dev, "test": test})
    for k, rows in pools.items():
        _write(out, f"pool{k}", {"train": rows})
    (out / "eval_keys.txt").write_text("\n".join(sorted(eval_keys)) + "\n")

    def n_agree(rows):
        return dict(sorted(Counter(r["n_agree"] for r in rows).items()))

    stats = {
        "scope_filter": {"max_tokens": max_tokens, "vocab": str(vocab)},
        "dropped_by_scope": {"train_published": dict(d_train), "test_published": dict(d_test)},
        "rows": {"train_all": len(train), "dev": len(dev), "test": len(test)},
        "articles": {
            "train_all": len({r["pmcid"] for r in train}),
            "dev": len({r["pmcid"] for r in dev}),
            "test": len({r["pmcid"] for r in test}),
        },
        "removed_from_train_for_overlap": n_train0 - len(train),
        "removed_from_dev_for_overlap": n_dev0 - len(dev),
        "n_agree": {"train_all": n_agree(train), "dev": n_agree(dev), "test": n_agree(test)},
        "pools": {
            str(k): {
                "rows": len(rows),
                "articles": len({r["pmcid"] for r in rows}),
                "n_agree": n_agree(rows),
            }
            for k, rows in pools.items()
        },
        "per_article_cap": per_article,
        "eval_connectivity_blocks": len(eval_keys),
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, default=Path("configs/vocab.json"))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--per-article", type=int, default=10)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.local, a.out, a.vocab, a.max_tokens, a.per_article), indent=1))


if __name__ == "__main__":
    main()
