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
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ouroboros.eval.standardize import standardize
from ouroboros.geometry.cache import calculator_weights_sha256
from ouroboros.geometry.conformers import (
    lowest_energy_conformer,
    mace_calculator,
    mirror_positions,
    relax,
)
from ouroboros.geometry.propagation import CATEGORIES, categorize, same_formula
from ouroboros.provenance import JsonCache, file_sha256, implementation_identity


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
    ap.add_argument("--cache", type=Path, help="default: <out>/energy-cache")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--skip-correct", action="store_true", help="dE = 0 by definition")
    ap.add_argument("--per-category", type=int, default=None, help="at most N pairs per category")
    ap.add_argument(
        "--noise-seeds",
        action="store_true",
        help="also search the truth's conformers with a second seed: conformer-search noise floor",
    )
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
    persistent = JsonCache(a.cache or out / "energy-cache")
    cache: dict[tuple, dict] = {}
    settings = {
        "schema": 1,
        "model": a.model,
        "weights": calculator_weights_sha256(calc),
        "n_conf": a.n_conf,
        "mmff_prescreen": a.mmff_prescreen,
        "fmax": 0.05,
        "steps": 500,
        "dtype": "float64",
        "device": str(calc.device),
        "versions": {name: version(name) for name in ("mace-torch", "rdkit", "ase", "torch")},
        "code": implementation_identity(Path(__file__).resolve().parents[1] / "ouroboros"),
        "script": file_sha256(__file__),
    }

    def energy(smi: str, seed: int = 0) -> dict:
        if (smi, seed) not in cache:

            def compute():
                g = lowest_energy_conformer(
                    smi, n_conf=a.n_conf, seed=seed, calc=calc, mmff_prescreen=a.mmff_prescreen
                )
                return {
                    "status": g.status,
                    "E": g.energy,
                    "converged": g.converged,
                    "sec": g.seconds,
                    "symbols": g.symbols,
                    "positions": g.positions.tolist() if g.positions is not None else None,
                    "failures": g.failures,
                    "embedding_attempts": g.embedding_attempts,
                }

            cache[(smi, seed)] = persistent.get(
                {**settings, "kind": "search", "smiles": smi, "seed": seed}, compute
            )
        return cache[(smi, seed)]

    def mirrored_energy(ref: dict, ref_smiles: str, pred_smiles: str) -> dict:
        """Enantiomer energy with the conformer search taken out: relax the mirror image of the
        truth's lowest-energy geometry exactly as the truth was relaxed."""
        from ase import Atoms

        atoms = Atoms(ref["symbols"], positions=mirror_positions(ref["positions"]))
        r = relax(atoms, calc)
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles(ref_smiles))
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for i, pos in enumerate(r.positions):
            conf.SetAtomPosition(i, pos.tolist())
        mol.AddConformer(conf, assignId=True)
        ok = r.converged and r.fmax <= 0.05 and np.isfinite(r.energy)
        ok = ok and np.isfinite(r.positions).all() and np.isfinite(r.forces).all()
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistryFrom3D(mol, confId=0, replaceExistingTags=True)
        expected = Chem.AddHs(Chem.MolFromSmiles(pred_smiles))
        ok = ok and mol.HasSubstructMatch(expected, useChirality=True)
        return {
            "status": "ok" if ok else "invalid_mirrored_geometry",
            "E": r.energy if ok else None,
            "converged": bool(ok),
        }

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
                if cat == "enantiomer" and er["status"] == "ok":
                    ep = persistent.get(
                        {
                            **settings,
                            "kind": "mirror",
                            "ref": rs,
                            "pred": ps,
                            "positions": er["positions"],
                        },
                        lambda er=er, rs=rs, ps=ps: mirrored_energy(er, rs, ps),
                    )
                    rec["pred_energy_method"] = "mirrored truth geometry"
                else:
                    ep = energy(ps) if ps != rs else er
                    rec["pred_energy_method"] = "independent conformer search"
                if a.noise_seeds and er["status"] == "ok":
                    e1 = energy(rs, seed=1)
                    if e1["E"] is not None:
                        rec["search_noise_dE"] = e1["E"] - er["E"]
                rec.update(
                    E_ref=er["E"],
                    E_pred=ep["E"],
                    status_ref=er["status"],
                    status_pred=ep["status"],
                    converged=bool(er["converged"] and ep["converged"]),
                )
                if (
                    rec["isomer"]
                    and rec["converged"]
                    and er["status"] == ep["status"] == "ok"
                    and er["E"] is not None
                    and ep["E"] is not None
                ):
                    rec["dE"] = ep["E"] - er["E"]
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
    noise = np.array([abs(r["search_noise_dE"]) for r in results if "search_noise_dE" in r])
    summary = {
        "conformer_search_noise_eV": (
            {"n": int(noise.size), "median": float(np.median(noise)), "max": float(noise.max())}
            if noise.size
            else None
        ),
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
