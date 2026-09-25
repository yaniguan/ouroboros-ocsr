"""Error propagation (Phase 6): energy of the predicted molecule vs. the ground truth, broken down
by recognition-error category.

Input: a predictions JSONL (``ref``, ``pred`` per line; e.g. ``<run>/eval/predictions.jsonl``,
angle-0 rows of one set) or ``--pairs-tsv`` with columns pred/ref. For every pair:
  category = correct | enantiomer | diastereomer | constitutional | invalid
  E_pred, E_ref = lowest relaxed MACE-OFF energies (N ETKDG conformers each, cached per SMILES)
  dE = E_pred - E_ref  (eV; only for isomers, i.e. same molecular formula — otherwise NaN)
Output: ``<out>/pairs.jsonl`` and ``<out>/summary.json`` (per category: count, #isomer pairs,
median / mean / max |dE|, embedding failures). By construction enantiomers give dE ≈ 0 — energy
cannot detect wrong chirality; stereo is scored separately.

    python scripts/error_propagation.py --predictions runs/X/eval/predictions.jsonl \
        --set rendered_test --out results/X_energy --n-conf 10 --model medium
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ouroboros.eval.standardize import standardize
from ouroboros.geometry.conformers import lowest_energy_conformer, mace_calculator
from ouroboros.geometry.propagation import CATEGORIES, categorize, same_formula


def load_pairs(a) -> list[dict]:
    if a.pairs_tsv:
        with open(a.pairs_tsv, newline="") as f:
            return [{"pred": r["pred"], "ref": r["ref"]} for r in csv.DictReader(f, delimiter="\t")]
    rows = [json.loads(x) for x in Path(a.predictions).read_text().splitlines()]
    rows = [r for r in rows if r.get("angle", 0.0) == 0.0 and (a.set is None or r["set"] == a.set)]
    pairs = [{"pred": r["pred"], "ref": r["ref"]} for r in rows]
    if a.per_category:  # balanced sample: first N pairs of every category (file order)
        seen: dict[str, int] = defaultdict(int)
        keep = []
        for p in pairs:
            c = categorize(p["pred"], p["ref"])
            if seen[c] < a.per_category:
                seen[c] += 1
                keep.append(p)
        pairs = keep
    return pairs[: a.max_pairs]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--predictions")
    ap.add_argument("--pairs-tsv")
    ap.add_argument("--set", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-conf", type=int, default=10)
    ap.add_argument("--model", default="small")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--skip-correct", action="store_true", help="dE = 0 by definition")
    ap.add_argument("--per-category", type=int, default=None, help="at most N pairs per category")
    ap.add_argument(
        "--mmff-prescreen",
        type=int,
        default=None,
        help="MMFF-relax all conformers, send only the k lowest to MACE",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calc = mace_calculator(a.model)
    cache: dict[str, dict] = {}

    def energy(smi: str) -> dict:
        if smi not in cache:
            g = lowest_energy_conformer(
                smi, n_conf=a.n_conf, seed=0, calc=calc, mmff_prescreen=a.mmff_prescreen
            )
            cache[smi] = {
                "status": g.status,
                "E": g.energy,
                "converged": g.converged,
                "sec": g.seconds,
            }
        return cache[smi]

    t0 = time.time()
    results = []
    with open(out / "pairs.jsonl", "w") as f:
        for p in load_pairs(a):
            cat = categorize(p["pred"], p["ref"])
            rec = {**p, "category": cat, "dE": float("nan"), "isomer": False}
            if cat not in ("invalid",) and not (cat == "correct" and a.skip_correct):
                ps, rs = standardize(p["pred"]).smiles, standardize(p["ref"]).smiles
                rec["isomer"] = same_formula(ps, rs)
                er = energy(rs)
                ep = energy(ps) if ps != rs else er
                rec.update(
                    E_ref=er["E"],
                    E_pred=ep["E"],
                    status_ref=er["status"],
                    status_pred=ep["status"],
                    converged=bool(er["converged"] and ep["converged"]),
                )
                if rec["isomer"] and er["E"] is not None and ep["E"] is not None:
                    rec["dE"] = ep["E"] - er["E"]
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
    summary = {
        "n_pairs": len(results),
        "model": f"MACE-OFF23 {a.model}",
        "n_conf": a.n_conf,
        "seconds": time.time() - t0,
        "categories": {},
    }
    by = defaultdict(list)
    for r in results:
        by[r["category"]].append(r)
    for cat in CATEGORIES:
        rs = by.get(cat, [])
        d = np.array([abs(r["dE"]) for r in rs if not math.isnan(r["dE"])])
        summary["categories"][cat] = {
            "n": len(rs),
            "n_isomer_pairs_with_energy": int(d.size),
            "median_abs_dE_eV": float(np.median(d)) if d.size else None,
            "mean_abs_dE_eV": float(d.mean()) if d.size else None,
            "max_abs_dE_eV": float(d.max()) if d.size else None,
            "embedding_failures": sum(
                1
                for r in rs
                if r.get("status_ref", "ok") != "ok" or r.get("status_pred", "ok") != "ok"
            ),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
