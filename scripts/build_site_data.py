"""Build site/assets/benchmark-data.json from committed Ouroboros benchmark receipts."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_rotation_arm(arm: str) -> dict[str, list]:
    raw = json.loads((ROOT / f"benchmarks/v0.1.0-alpha/{arm}-summary.json").read_text())
    sweep = [e for e in raw["entries"] if e["set"] == "rendered_test:sweep"]
    sweep.sort(key=lambda e: e["angle"])
    return {
        "angles": [int(round(e["angle"])) for e in sweep],
        "exact": [round(e["exact"] * 100, 2) for e in sweep],
        "exact_ci": [
            [round(e["exact_ci95"][0] * 100, 2), round(e["exact_ci95"][1] * 100, 2)]
            for e in sweep
        ],
        "tanimoto": [round(e["tanimoto"] * 100, 2) for e in sweep],
        "stereo": [round(e["exact_on_stereo_subset"] * 100, 2) for e in sweep],
    }


def _parse_md_report(path: Path) -> dict[str, dict[str, float | list[float]]]:
    out: dict[str, dict[str, float | list[float]]] = {}
    pat = re.compile(
        r"^\|\s*\*?\*?([A-Za-z_]+)\*?\*?\s*\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*"
        r"\[([\d.]+)%,\s*([\d.]+)%\]\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*"
        r"([\d.]+)%\s*\|\s*([\d.]+)\s*\|"
    )
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            sub = m.group(1)
            out[sub] = {
                "n": int(m.group(2)),
                "exact": float(m.group(3)),
                "ci": [float(m.group(4)), float(m.group(5))],
                "exact_flat": float(m.group(6)),
                "inchikey": float(m.group(7)),
                "valid": float(m.group(8)),
                "tanimoto": round(float(m.group(9)) * 100, 2),
            }
    return out


def build_data() -> dict:
    arm_a = _load_rotation_arm("A")
    arm_b = _load_rotation_arm("B")
    arm_c = _load_rotation_arm("C")

    ft_dir = ROOT / "benchmarks/finetune-v2-oct30"
    dev_ens = _parse_md_report(ft_dir / "AC-ensemble-beam5-ttapad-report.md")
    frozen_ens = _parse_md_report(
        ROOT / "benchmarks/frozen-test-v2/AC-ensemble-beam5-ttapad-report.md"
    )
    hd_ens = _parse_md_report(
        ROOT / "benchmarks/handdrawn-decimer/AC-ensemble-beam5-ttapad-report.md"
    )
    real_models = [
        (
            "AC ensemble (beam 5 + TTA)",
            "#b5d88e",
            dev_ens,
        ),
        (
            "AC ensemble (beam 5)",
            "#79c2bd",
            _parse_md_report(ft_dir / "AC-ensemble-beam5-report.md"),
        ),
        (
            "Arm A fine-tuned (greedy)",
            "#91b6df",
            _parse_md_report(ft_dir / "A-selected-report.md"),
        ),
        (
            "Arm C fine-tuned (greedy)",
            "#b9a1de",
            _parse_md_report(ft_dir / "C-selected-report.md"),
        ),
    ]
    a_v0 = json.loads((ROOT / "benchmarks/real-subset100-rgroups/A_report.json").read_text())
    v0_map = {}
    for g in a_v0["groups"]:
        v0_map[g["subset"]] = {
            "n": g["n"],
            "exact": round(g["exact"] * 100, 2),
            "ci": [round(g["exact_ci95"][0] * 100, 2), round(g["exact_ci95"][1] * 100, 2)],
        }
    real_models.append(("Arm A synthetic zero-shot", "#d97757", v0_map))

    # P1 ChemVeritas-600 measured summary (tag p1-v1.0 on branch p1-phys-spec-ocsr-bench)
    p1_ablations = [
        {
            "id": "A0",
            "label": "Visual top-1",
            "t2_exact": 19.67,
            "t2_ci": [16.50, 23.17],
            "rpf": 46.77,
            "t3_audit": 50.00,
            "t5_abstain": 0.0,
            "t5_hallucinated": 100.0,
        },
        {
            "id": "A1",
            "label": "+ RDKit valence",
            "t2_exact": 19.67,
            "t2_ci": [16.50, 23.17],
            "rpf": 46.77,
            "t3_audit": 50.00,
            "t5_abstain": 100.0,
            "t5_hallucinated": 0.0,
        },
        {
            "id": "A2",
            "label": "+ HRMS formula",
            "t2_exact": 23.67,
            "t2_ci": [20.33, 27.17],
            "rpf": 50.62,
            "t3_audit": 50.00,
            "t5_abstain": 100.0,
            "t5_hallucinated": 0.0,
        },
        {
            "id": "A3",
            "label": "+ 3D MACE strain",
            "t2_exact": 23.33,
            "t2_ci": [20.17, 27.17],
            "rpf": 54.83,
            "t3_audit": 45.67,
            "t5_abstain": 100.0,
            "t5_hallucinated": 0.0,
        },
        {
            "id": "A4",
            "label": "+ 13C NMR orbits",
            "t2_exact": 77.83,
            "t2_ci": [74.67, 81.17],
            "rpf": 93.46,
            "t3_audit": 61.00,
            "t5_abstain": 100.0,
            "t5_hallucinated": 0.0,
        },
        {
            "id": "A5",
            "label": "+ 1H & Karplus 3J",
            "t2_exact": 77.00,
            "t2_ci": [73.83, 80.33],
            "rpf": 93.25,
            "t3_audit": 61.00,
            "t5_abstain": 100.0,
            "t5_hallucinated": 0.0,
        },
        {
            "id": "A6",
            "label": "Full Bayesian loop",
            "t2_exact": 77.00,
            "t2_ci": [73.83, 80.33],
            "rpf": 93.25,
            "t3_audit": 74.83,
            "t5_abstain": 100.0,
            "t5_hallucinated": 0.0,
        },
    ]

    p1_closed_loop_views = [
        {"view": "V0 · Clean upright", "visual": 1.0, "bayesian": 14.0, "ci": [7.0, 21.0]},
        {"view": "V1 · Off-grid rotated", "visual": 1.0, "bayesian": 12.0, "ci": [6.0, 19.0]},
        {"view": "V2 · Crowded SI scan", "visual": 1.0, "bayesian": 3.0, "ci": [0.0, 7.0]},
        {"view": "V3 · Superatom layout", "visual": 0.0, "bayesian": 4.0, "ci": [1.0, 8.0]},
    ]

    return {
        "rotation_sweep": {
            "angles": arm_c["angles"],
            "models": [
                {
                    "id": "C",
                    "name": "Arm C · C8 steerable CNN (32.4M, no aug)",
                    "color": "#b5d88e",
                    "width": 3.4,
                    "dash": [],
                    "exact": arm_c["exact"],
                    "exact_ci": arm_c["exact_ci"],
                    "tanimoto": arm_c["tanimoto"],
                    "stereo": arm_c["stereo"],
                },
                {
                    "id": "B",
                    "name": "Arm B · CNN + 360° rotation aug (43.2M)",
                    "color": "#91b6df",
                    "width": 2.3,
                    "dash": [7, 5],
                    "exact": arm_b["exact"],
                    "exact_ci": arm_b["exact_ci"],
                    "tanimoto": arm_b["tanimoto"],
                    "stereo": arm_b["stereo"],
                },
                {
                    "id": "A",
                    "name": "Arm A · Plain CNN baseline (43.2M, no aug)",
                    "color": "#d97757",
                    "width": 2.3,
                    "dash": [],
                    "exact": arm_a["exact"],
                    "exact_ci": arm_a["exact_ci"],
                    "tanimoto": arm_a["tanimoto"],
                    "stereo": arm_a["stereo"],
                },
            ],
        },
        "real_benchmarks": {
            "subsets": ["ALL", "UOB", "ACS", "USPTO"],
            "subset_labels": {
                "ALL": "All figures (n = 298)",
                "UOB": "UOB (n = 100)",
                "ACS": "ACS journals (n = 100)",
                "USPTO": "USPTO patents (n = 98)",
            },
            "models": [
                {
                    "name": name,
                    "color": color,
                    "scores": {s: data[s]["exact"] for s in ("ALL", "UOB", "ACS", "USPTO")},
                    "cis": {s: data[s]["ci"] for s in ("ALL", "UOB", "ACS", "USPTO")},
                }
                for name, color, data in real_models
            ],
        },
        "frozen_vs_dev": {
            "groups": [
                {
                    "id": "ALL",
                    "label": "All figures",
                    "frozen": {"exact": frozen_ens["ALL"]["exact"], "ci": frozen_ens["ALL"]["ci"]},
                    "dev": {"exact": dev_ens["ALL"]["exact"], "ci": dev_ens["ALL"]["ci"]},
                    "handdrawn": {"exact": hd_ens["ALL"]["exact"], "ci": hd_ens["ALL"]["ci"]},
                },
                {
                    "id": "UOB",
                    "label": "UOB",
                    "frozen": {"exact": frozen_ens["UOB"]["exact"], "ci": frozen_ens["UOB"]["ci"]},
                    "dev": {"exact": dev_ens["UOB"]["exact"], "ci": dev_ens["UOB"]["ci"]},
                    "handdrawn": None,
                },
                {
                    "id": "ACS",
                    "label": "ACS journals",
                    "frozen": {"exact": frozen_ens["ACS"]["exact"], "ci": frozen_ens["ACS"]["ci"]},
                    "dev": {"exact": dev_ens["ACS"]["exact"], "ci": dev_ens["ACS"]["ci"]},
                    "handdrawn": None,
                },
                {
                    "id": "USPTO",
                    "label": "USPTO patents",
                    "frozen": {
                        "exact": frozen_ens["USPTO"]["exact"],
                        "ci": frozen_ens["USPTO"]["ci"],
                    },
                    "dev": {"exact": dev_ens["USPTO"]["exact"], "ci": dev_ens["USPTO"]["ci"]},
                    "handdrawn": None,
                },
                {
                    "id": "no_stereo",
                    "label": "No-stereo subset",
                    "frozen": {
                        "exact": frozen_ens["no_stereo"]["exact"],
                        "ci": frozen_ens["no_stereo"]["ci"],
                    },
                    "dev": {
                        "exact": dev_ens["no_stereo"]["exact"],
                        "ci": dev_ens["no_stereo"]["ci"],
                    },
                    "handdrawn": {
                        "exact": hd_ens["no_stereo"]["exact"],
                        "ci": hd_ens["no_stereo"]["ci"],
                    },
                },
                {
                    "id": "has_stereo",
                    "label": "Stereo subset",
                    "frozen": {
                        "exact": frozen_ens["has_stereo"]["exact"],
                        "ci": frozen_ens["has_stereo"]["ci"],
                    },
                    "dev": {
                        "exact": dev_ens["has_stereo"]["exact"],
                        "ci": dev_ens["has_stereo"]["ci"],
                    },
                    "handdrawn": {
                        "exact": hd_ens["has_stereo"]["exact"],
                        "ci": hd_ens["has_stereo"]["ci"],
                    },
                },
            ]
        },
        "p1_chemveritas": {
            "ablations": p1_ablations,
            "closed_loop_views": p1_closed_loop_views,
            "strain_audit": {
                "n_pairs": 50,
                "n_embed_failures": 30,
                "n_measured_pairs": 20,
                "measured_median_mace_kcal": 1.97,
                "measured_median_mmff_kcal": 2.60,
                "measured_pref_mace": 80.0,
                "measured_pref_mmff": 100.0,
                "pooled_pref": 92.0,
            },
            "literature_nmr": {
                "n_compounds": 30,
                "n_pairs": 15,
                "n_signals": 181,
                "c13_mae_ppm": 5.25,
                "pearson_r": 0.981,
                "recovery_mmff": 73.3,
                "recovery_mace": 66.7,
            },
        },
    }


def main() -> None:
    dst = ROOT / "site/assets/benchmark-data.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(build_data(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
