"""Bayesian physics + spectroscopy OCSR reranker CLI (ablations A0-A6).

Reads JSONL rows containing ``candidates`` (list of ``[smiles, logprob]`` or ``smiles`` strings),
optional ``si`` dict or ``si_text`` experimental paragraph, and optional ``truth_smiles``;
outputs reranked predictions JSONL and summary JSON with bootstrap 95% CIs, ECE, and AURC.

Example:
    python scripts/rerank_phys_spec.py --input candidates.jsonl --out results/rerank --ablation all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ouroboros.bench.scoring import bootstrap_ci
from ouroboros.eval.standardize import flat_smiles, standardize
from ouroboros.phys.posterior import (
    ABLATION_PRESETS,
    SIObservables,
    area_under_risk_coverage,
    expected_calibration_error,
    rerank_candidates,
)
from ouroboros.si.parser import SICharacterization, parse_si_block


def _parse_candidates(raw_cands: list) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for idx, item in enumerate(raw_cands):
        if isinstance(item, str):
            out.append((item, -0.05 * idx))
        elif isinstance(item, list | tuple) and len(item) >= 2:
            out.append((str(item[0]), float(item[1])))
        elif isinstance(item, dict):
            out.append((str(item["smiles"]), float(item.get("logprob", -0.05 * idx))))
    return out


def _extract_observables(row: dict) -> SIObservables:
    if "si" in row and isinstance(row["si"], dict):
        return SICharacterization.from_dict(row["si"]).to_observables()
    if "si_text" in row and row["si_text"]:
        return parse_si_block(str(row["si_text"])).to_observables()
    return SIObservables()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Input candidates JSONL path")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--ablation",
        choices=(*ABLATION_PRESETS.keys(), "all"),
        default="A6_full_posterior",
        help="Posterior ablation preset or 'all'",
    )
    p.add_argument("--n-conf", type=int, default=4, help="3D conformers per candidate")
    p.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed")
    p.add_argument("--mace-model", type=str, default=None, help="Optional MACE-OFF model size")
    args = p.parse_args(argv)

    calc = None
    if args.mace_model:
        from ouroboros.geometry.conformers import mace_calculator

        calc = mace_calculator(args.mace_model)

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ablations = list(ABLATION_PRESETS) if args.ablation == "all" else [args.ablation]
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for abl in ablations:
        out_rows: list[dict] = []
        exact_vec, flat_vec, abstain_vec, confs = [], [], [], []
        for idx, row in enumerate(rows):
            cands = _parse_candidates(row["candidates"])
            obs = _extract_observables(row)
            dec = rerank_candidates(
                cands, obs, ablation=abl, n_conf=args.n_conf, seed=args.seed, calc=calc
            )
            rec = {
                "id": row.get("id", row.get("task_id", idx)),
                "ablation": abl,
                "selected_smiles": dec.selected_smiles,
                "canonical_smiles": dec.canonical_smiles,
                "posterior_prob": dec.posterior_prob,
                "margin": dec.margin,
                "epistemic_status": dec.epistemic_status,
                "candidates": [asdict(c) for c in dec.candidates],
            }
            abstain_vec.append(
                1.0 if dec.epistemic_status == "abstain_enantiomer_underdetermined" else 0.0
            )
            if "truth_smiles" in row and row["truth_smiles"]:
                st_p = standardize(dec.selected_smiles)
                st_t = standardize(row["truth_smiles"])
                exact_vec.append(1.0 if (st_p.valid and st_p.smiles == st_t.smiles) else 0.0)
                flat_vec.append(
                    1.0 if (st_p.valid and flat_smiles(st_p) == flat_smiles(st_t)) else 0.0
                )
                confs.append(dec.posterior_prob)
            out_rows.append(rec)

        pred_path = args.out / f"predictions_{abl}.jsonl"
        pred_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in out_rows), encoding="utf-8"
        )
        abl_report: dict[str, object] = {
            "n_items": len(out_rows),
            "parity_abstain_rate": bootstrap_ci(abstain_vec, seed=args.seed),
        }
        if exact_vec:
            bool_exact = [bool(x) for x in exact_vec]
            abl_report.update(
                {
                    "stereo_exact_match": bootstrap_ci(exact_vec, seed=args.seed),
                    "flat_connectivity_match": bootstrap_ci(flat_vec, seed=args.seed),
                    "ece": expected_calibration_error(confs, bool_exact),
                    "aurc": area_under_risk_coverage(confs, bool_exact),
                }
            )
        summary[abl] = abl_report

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
