"""Phase 6 benchmarks on filtered test molecules (writes benchmarks/geometry/*.json).

python scripts/bench_geometry.py embed   --n 1000              # ETKDG success + stereo
python scripts/bench_geometry.py relax   --n 100 --n-conf 3    # MACE-OFF convergence, time
python scripts/bench_geometry.py enantio --n 20                # enantiomer ΔE
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

from ouroboros.data.build import read_manifest
from ouroboros.geometry.conformers import (
    FMAX,
    embed,
    lowest_energy_conformer,
    mace_calculator,
    mirror_positions,
    relax,
    stereo_preserved,
    to_atoms,
)
from ouroboros.geometry.propagation import mirror_smiles

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "geometry"


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    obj.update(date=time.strftime("%Y-%m-%d"), host=platform.node())
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(json.dumps({k: v for k, v in obj.items() if k != "failures"}, indent=1))


def cmd_embed(a, rows):
    fails, stereo_total, stereo_ok, n_ok = [], 0, 0, 0
    chiral = [r for r in rows if "@" in r["smiles"] or "/" in r["smiles"] or "\\" in r["smiles"]]
    t0 = time.perf_counter()
    for r in rows[: a.n]:
        mol, why = embed(r["smiles"], a.n_conf, seed=0)
        if mol is None:
            fails.append({"smiles": r["smiles"], "reason": why})
            continue
        n_ok += 1
    t_embed = time.perf_counter() - t0
    for r in chiral[: a.n]:
        mol, why = embed(r["smiles"], a.n_conf, seed=0)
        if mol is None:
            continue
        stereo_total += 1
        stereo_ok += all(stereo_preserved(mol, r["smiles"]))
    _save(
        "embed",
        {
            "n": min(a.n, len(rows)),
            "n_conf": a.n_conf,
            "success": n_ok,
            "success_rate": n_ok / min(a.n, len(rows)),
            "failure_reasons": dict(Counter(f["reason"] for f in fails)),
            "failures": fails,
            "stereo_n": stereo_total,
            "stereo_all_conformers_ok": stereo_ok,
            "stereo_rate": stereo_ok / max(1, stereo_total),
            "embed_sec_per_mol": t_embed / min(a.n, len(rows)),
        },
    )


def cmd_relax(a, rows):
    calc = mace_calculator(a.model)
    res = []
    for r in rows[: a.n]:
        g = lowest_energy_conformer(r["smiles"], n_conf=a.n_conf, seed=0, calc=calc)
        res.append(g)
        print(r["smiles"], g.status, g.converged, g.all_converged, f"{g.seconds:.1f}s", flush=True)
    ok = [g for g in res if g.status == "ok"]
    _save(
        "relax",
        {
            "model": f"MACE-OFF23 {a.model}",
            "n": len(res),
            "n_conf": a.n_conf,
            "fmax": FMAX,
            "embedded": len(ok),
            "lowest_converged_rate": sum(g.converged for g in ok) / max(1, len(ok)),
            "all_conformers_converged_rate": sum(g.all_converged for g in ok) / max(1, len(ok)),
            "stereo_preserved_after_relax": sum(bool(g.stereo_ok_final) for g in ok)
            / max(1, len(ok)),
            "median_sec_per_mol": statistics.median(g.seconds for g in ok) if ok else None,
            "device": str(getattr(calc, "device", "")),
        },
    )


def cmd_enantio(a, rows):
    calc = mace_calculator(a.model)
    chiral = [r["smiles"] for r in rows if "@" in r["smiles"]]
    chiral = [s for s in chiral if Chem.CanonSmiles(mirror_smiles(s)) != Chem.CanonSmiles(s)]
    pairs, skipped = [], []
    for s in chiral:
        if len(pairs) >= a.n:
            break
        mol, why = embed(s, 1, seed=0)
        if mol is None:
            skipped.append({"smiles": s, "reason": why})
            continue
        atoms = to_atoms(mol, mol.GetConformers()[0].GetId())
        mirrored = atoms.copy()
        mirrored.set_positions(mirror_positions(atoms.get_positions()))
        r1, r2 = relax(atoms, calc), relax(mirrored, calc)
        pairs.append(
            {
                "smiles": s,
                "enantiomer": Chem.CanonSmiles(mirror_smiles(s)),
                "E": r1.energy,
                "E_mirror": r2.energy,
                "dE": abs(r1.energy - r2.energy),
                "converged": r1.converged and r2.converged,
            }
        )
        print(pairs[-1], flush=True)
    dE = np.array([p["dE"] for p in pairs])
    _save(
        "enantiomers",
        {
            "n_pairs": len(pairs),
            "max_abs_dE_eV": float(dE.max()),
            "pass": bool(len(pairs) >= 20 and dE.max() < 1e-3),
            "pairs": pairs,
            "skipped_not_embeddable": skipped,
            "protocol": "mirror (x -> -x) of the same ETKDG conformer, identical "
            "LBFGS relaxation with MACE-OFF, fmax 0.05 eV/A",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "relax", "enantio"])
    ap.add_argument("--manifest", default="data/full/manifests/test.tsv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-conf", type=int, default=10)
    ap.add_argument("--model", default="small")
    a = ap.parse_args()
    rows = read_manifest(a.manifest)
    {"embed": cmd_embed, "relax": cmd_relax, "enantio": cmd_enantio}[a.cmd](a, rows)


if __name__ == "__main__":
    main()
