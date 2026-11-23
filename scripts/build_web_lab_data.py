"""Precompute compact, real WorkbenchEngine snapshots for the static GitHub Pages 3D Lab."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ouroboros.ui import WorkbenchEngine, audit_3d_physical_compliance  # noqa: E402


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 8) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def build_lab_presets() -> dict[str, Any]:
    torch.set_num_threads(2)
    eng = WorkbenchEngine(device="cpu")

    c3d_specs = [
        ("ala", "L-Alanine (chiral amino acid)", "N[C@@H](C)C(=O)O", "mmff94s", False),
        (
            "ibu",
            "(S)-Ibuprofen (chiral NSAID)",
            "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
            "mmff94s",
            False,
        ),
        ("resv", "(E)-Resveratrol (E-alkene)", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1", "mmff94s", False),
        (
            "caffeine",
            "Caffeine (purine alkaloid)",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "mmff94s",
            False,
        ),
        (
            "camphor",
            "(+)-Camphor (P1 bridged cage)",
            "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2",
            "mmff94s",
            False,
        ),
        ("bcp", "BCP [1.1.1] amide (P2 bioisostere)", "CNC(=O)C12CC(C1)C2", "mmff94s", False),
        (
            "bch",
            "Bicyclo[2.1.1]hexane amide (P2)",
            "NC(=O)[C@]12CC[C@H](C1)C2",
            "mmff94s",
            False,
        ),
        ("cubane", "Cubane amide (P2 90° cage)", "NC(=O)C12C3C4C1C5C2C3C45", "mmff94s", False),
        ("spiro", "Spiro-oxetane amide (P2)", "O=C(NC1CC2(C1)COC2)c1ccccc1", "mmff94s", False),
        ("alias", "[Boc] & [Ph] shorthand expansion", "[Boc]NCc1ccc([Ph])cc1", "mmff94s", False),
        ("ec", "Ethylene carbonate · EC (electrolyte)", "O=C1OCCO1", "mmff94s", False),
        (
            "fec",
            "(R)-Fluoroethylene carbonate · FEC (SEI additive)",
            "O=C1OC[C@H](F)O1",
            "mmff94s",
            False,
        ),
        ("vc", "Vinylene carbonate · VC (SEI additive)", "O=c1occo1", "mmff94s", False),
        ("dme", "1,2-Dimethoxyethane · DME (glyme solvent)", "COCCOC", "mmff94s", False),
        (
            "tte",
            "TTE fluorinated ether · LHCE diluent",
            "FC(F)C(F)(F)OCC(F)(F)F",
            "mmff94s",
            False,
        ),
    ]

    compliance_presets = []
    for key, label, smi, pot, clash in c3d_specs:
        res = audit_3d_physical_compliance(
            smi,
            potential=pot,
            n_conf=3 if not clash else 1,
            do_relax=True,
            steps=250,
            expand_abbrev=True,
            inject_steric_clash=clash,
        )
        compliance_presets.append(
            {
                "key": key,
                "label": label,
                "smiles": smi,
                "potential": pot,
                "clash": clash,
                "result": _clean(res),
            }
        )

    eq_cases = {}
    for arch_key, arch, order in [
        ("steerable_c8", "steerable", 8),
        ("equiv_attn_c4", "equiv_attention", 4),
        ("baseline_cnn", "baseline", 8),
    ]:
        eq_cases[arch_key] = {}
        for ang in (0, 30, 45, 90, 180, 270):
            r = eng.run_2d_equivariance(
                {
                    "mode": "arch",
                    "arch": arch,
                    "order": order,
                    "smiles": "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
                    "angle_deg": float(ang),
                }
            )
            eq_cases[arch_key][str(ang)] = _clean(r)

    prop_menthol = _clean(
        eng.run_error_propagation(
            {
                "truth_smiles": "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1",
                "potential": "mmff94s",
                "n_conf": 6,
                "relax": True,
            }
        )
    )
    prop_camphor = _clean(
        eng.run_error_propagation(
            {
                "truth_smiles": "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2",
                "potential": "mmff94s",
                "n_conf": 6,
                "relax": True,
            }
        )
    )

    return {
        "compliance": compliance_presets,
        "equivariance2d": eq_cases,
        "propagation": {
            "menthol": prop_menthol,
            "camphor": prop_camphor,
        },
    }


def main() -> None:
    dst = ROOT / "site/assets/lab-presets.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = build_lab_presets()
    dst.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
