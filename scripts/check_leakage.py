"""Leakage check: InChIKey overlap between every eval set and every training set (Am1-A).

A set is given as ``NAME=SPEC`` where SPEC is one of
  * ``path/to/manifest.tsv[:N]``   synthetic manifest (``smiles`` column), optional first-N prefix
  * ``path/to/shards_dir:SPLIT[:N]`` WebDataset shards (synthetic or real), optional prefix

Prints a table of overlap counts (full standard InChIKey, and connectivity block for
information) and exits with status 1 if ANY full-InChIKey overlap is > 0.

    python scripts/check_leakage.py \
        --eval  uspto=real/uspto:test  acs=real/acs:test  rendered=data/full/shards:test \
        --train synth1m=data/full/manifests/train.tsv  uspto_train=real/uspto:train

``--dump-eval-keys FILE`` writes the eval sets' InChIKeys, to be passed to
``scripts/build_dataset.py --exclude-keys FILE`` so the synthetic splits never contain them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from rdkit import RDLogger

from ouroboros.eval.standardize import standard_inchikey, standardize

RDLogger.DisableLog("rdApp.*")


def _key(smiles: str) -> str | None:
    return standard_inchikey(standardize(smiles))


def load_keys(spec: str) -> set[str]:
    parts = spec.split(":")
    path = Path(parts[0])
    if path.suffix == ".tsv":
        limit = int(parts[1]) if len(parts) > 1 else None
        with open(path, newline="") as f:
            smiles = [r["smiles"] for r in csv.DictReader(f, delimiter="\t")]
        smiles = smiles[:limit] if limit else smiles
    else:
        from ouroboros.data.loader import ShardDataset

        split = parts[1] if len(parts) > 1 else "test"
        limit = int(parts[2]) if len(parts) > 2 else None
        ds = ShardDataset(path, split, max_samples=limit)
        smiles = [ds.meta(i)["smiles"] for i in range(len(ds))]
    keys = {_key(s) for s in smiles}
    keys.discard(None)
    return keys


def overlap_table(evals: dict[str, set[str]], trains: dict[str, set[str]]) -> list[dict]:
    rows = []
    for en, ek in evals.items():
        e14 = {k[:14] for k in ek}
        for tn, tk in trains.items():
            rows.append(
                {
                    "eval": en,
                    "train": tn,
                    "n_eval": len(ek),
                    "n_train": len(tk),
                    "overlap_inchikey": len(ek & tk),
                    "overlap_connectivity": len(e14 & {k[:14] for k in tk}),
                }
            )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval", nargs="+", required=True, metavar="NAME=SPEC")
    ap.add_argument("--train", nargs="*", default=[], metavar="NAME=SPEC")
    ap.add_argument("--dump-eval-keys", default=None)
    a = ap.parse_args(argv)
    evals = {s.split("=", 1)[0]: load_keys(s.split("=", 1)[1]) for s in a.eval}
    trains = {s.split("=", 1)[0]: load_keys(s.split("=", 1)[1]) for s in a.train}
    if a.dump_eval_keys:
        keys = sorted(set().union(*evals.values()))
        Path(a.dump_eval_keys).write_text("\n".join(keys) + "\n")
        print(f"wrote {len(keys)} eval InChIKeys to {a.dump_eval_keys}")
    rows = overlap_table(evals, trains)
    hdr = f"{'eval':<16}{'train':<16}{'n_eval':>9}{'n_train':>10}{'InChIKey':>10}{'connect.':>10}"
    print(hdr)
    for r in rows:
        print(
            f"{r['eval']:<16}{r['train']:<16}{r['n_eval']:>9}{r['n_train']:>10}"
            f"{r['overlap_inchikey']:>10}{r['overlap_connectivity']:>10}"
        )
    bad = [r for r in rows if r["overlap_inchikey"] > 0]
    if bad:
        print(f"LEAKAGE: {len(bad)} eval/train pair(s) share InChIKeys", file=sys.stderr)
        return 1
    print("OK: 0 InChIKey overlap for every eval/train pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
