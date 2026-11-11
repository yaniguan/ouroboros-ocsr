"""Scan medicinal-chemistry SMILES or bioisostere panels for MLIP force blind spots.

Applies MACE_OFF23_POLICY (neutral closed-shell 10-element gate), detects 7 post-2020 3D
bioisostere families, generates exit-vector torsion + rattle geometries, evaluates MLIP
committee force disagreement sigma_F(R), and outputs stratified summary statistics.

Examples:
    python scripts/scan_mlip_blindspots.py --preset smoke --out results/blindspot_scan
    python scripts/scan_mlip_blindspots.py --input-smiles preds.jsonl --out results/lit_scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem

from ouroboros.active.committee import evaluate_committee
from ouroboros.active.distill import bootstrap_ci, mann_whitney_u_pvalue
from ouroboros.active.motifs import (
    BioisostereRecord,
    build_bioisostere_panel,
    check_mace_off23_eligibility,
    detect_bioisostere_families,
)
from ouroboros.active.oracle import build_pretrained_baseline_committee
from ouroboros.active.torsion import generate_blindspot_frames
from ouroboros.geometry.splits import connectivity_key


def _load_custom_smiles(path: Path) -> tuple[list[BioisostereRecord], list[dict]]:
    eligible_recs: list[BioisostereRecord] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for idx, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        s = raw_line.strip()
        if not s:
            continue
        if s.startswith("{"):
            obj = json.loads(s)
            smi_in = str(obj.get("smiles") or obj.get("pred") or "")
            mid = str(obj.get("id") or obj.get("mol_id") or f"LIT-{idx:04d}")
        else:
            smi_in = s.split()[0]
            mid = f"LIT-{idx:04d}"

        ok, canon, reasons = check_mace_off23_eligibility(smi_in)
        if not ok or canon is None:
            rejected.append({"mol_id": mid, "raw_smiles": smi_in, "reasons": list(reasons)})
            continue
        ck = connectivity_key(canon)
        if ck in seen:
            continue
        seen.add(ck)
        fams = detect_bioisostere_families(canon)
        grp = "bioisostere" if fams else "control"
        fam = fams[0] if fams else "planar_control"
        mol = Chem.MolFromSmiles(canon)
        eligible_recs.append(
            BioisostereRecord(
                mol_id=mid,
                smiles=canon,
                connectivity_key=ck,
                group=grp,
                family=fam,
                paired_id=None,
                n_heavy_atoms=mol.GetNumHeavyAtoms(),
                eligible=True,
                rejection_reasons=(),
            )
        )
    return eligible_recs, rejected


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    p.add_argument(
        "--input-smiles",
        type=Path,
        default=None,
        help="Optional JSONL/text file of OCSR-extracted SMILES (overrides --preset)",
    )
    p.add_argument("--torsion-step-deg", type=float, default=60.0)
    p.add_argument("--n-displaced", type=int, default=1)
    p.add_argument("--mace-models", nargs="*", default=(), help="Optional MACE-OFF sizes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    args = p.parse_args(argv)

    if args.input_smiles is not None:
        records, rejected = _load_custom_smiles(args.input_smiles)
    else:
        records = build_bioisostere_panel(args.preset)
        rejected = []

    frames = []
    for rec in records:
        frames.extend(
            generate_blindspot_frames(
                rec,
                torsion_step_deg=args.torsion_step_deg,
                n_displaced=args.n_displaced,
                seed=args.seed,
            )
        )

    committee = build_pretrained_baseline_committee(args.mace_models)
    scores = evaluate_committee(frames, committee)

    args.out.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    by_family: dict[str, list[float]] = {}
    for fr, sc in zip(frames, scores, strict=True):
        by_family.setdefault(sc.family, []).append(sc.sigma_f_max)
        frame_rows.append(
            {
                "frame_id": fr.frame_id,
                "mol_id": fr.mol_id,
                "smiles": fr.smiles,
                "group": fr.group,
                "family": fr.family,
                "source": fr.source,
                "dihedral_deg": fr.dihedral_deg,
                "transannular_c1_c3_a": fr.transannular_c1_c3_a,
                "sigma_f_max": sc.sigma_f_max,
                "sigma_f_mean": sc.sigma_f_mean,
                "rel_energy_std_kcal": sc.rel_energy_std_kcal,
            }
        )

    (args.out / "blindspot_frames.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in frame_rows), encoding="utf-8"
    )

    bio_sf = [s.sigma_f_max for s in scores if s.group == "bioisostere"]
    ctl_sf = [s.sigma_f_max for s in scores if s.group == "control"]
    summary = {
        "n_eligible_scaffolds": len(records),
        "n_rejected_ineligible": len(rejected),
        "rejected_records": rejected,
        "n_frames_evaluated": len(frames),
        "bioisostere_sigma_f_max": bootstrap_ci(bio_sf, seed=args.seed),
        "control_sigma_f_max": bootstrap_ci(ctl_sf, seed=args.seed),
        "mann_whitney_p_value": mann_whitney_u_pvalue(bio_sf, ctl_sf),
        "per_family_sigma_f_max": {
            fam: bootstrap_ci(vals, seed=args.seed) for fam, vals in sorted(by_family.items())
        },
    }
    (args.out / "scan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
