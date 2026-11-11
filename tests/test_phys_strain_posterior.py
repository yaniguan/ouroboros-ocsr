"""Tests for O(3) parity-invariant 3D strain, Bayesian posterior ablations, and CLI reranking."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from ouroboros.geometry.propagation import mirror_smiles
from ouroboros.phys.nmr import predict_nmr, score_nmr
from ouroboros.phys.posterior import (
    area_under_risk_coverage,
    expected_calibration_error,
    rerank_candidates,
)
from ouroboros.phys.strain import evaluate_3d_strain
from ouroboros.si.parser import format_si_paragraph
from scripts.rerank_phys_spec import main as rerank_cli_main


def test_h1_3_enantiomeric_parity_invariance_and_abstention():
    """Hypothesis H1.3 negative control: achiral physics & NMR are strictly O(3)-even."""
    l_menthol = Chem.CanonSmiles("C[C@@H]1CC[C@H](C(C)C)[C@H](O)C1")
    d_menthol = Chem.CanonSmiles(mirror_smiles(l_menthol))
    assert l_menthol != d_menthol

    s_l = evaluate_3d_strain(l_menthol, n_conf=6, seed=42)
    s_d = evaluate_3d_strain(d_menthol, n_conf=6, seed=42)
    assert s_l.min_energy_kcal is not None and s_d.min_energy_kcal is not None
    assert abs(s_l.min_energy_kcal - s_d.min_energy_kcal) < 1e-9

    nmr_l = predict_nmr(l_menthol, strain=s_l)
    nmr_d = predict_nmr(d_menthol, strain=s_d)
    m_l = score_nmr(l_menthol, obs_c13_ppm=nmr_l.c13_shifts_ppm, strain=s_l)
    m_d = score_nmr(d_menthol, obs_c13_ppm=nmr_l.c13_shifts_ppm, strain=s_d)
    assert abs((m_l.c13_mae_ppm or 0.0) - (m_d.c13_mae_ppm or 0.0)) < 1e-9
    assert nmr_d.c13_sym_count == nmr_l.c13_sym_count

    mass = float(Descriptors.ExactMolWt(Chem.AddHs(Chem.MolFromSmiles(l_menthol))))
    si, _ = format_si_paragraph("menthol", formula="C10H20O", neutral_mass=mass, nmr=nmr_l)

    # When visual priors are tied and [alpha]_D is absent, posterior MUST abstain on parity!
    dec_tied = rerank_candidates(
        [(l_menthol, -1.20), (d_menthol, -1.21)],
        si.to_observables(),
        ablation="A6_full_posterior",
        n_conf=4,
        seed=42,
    )
    assert dec_tied.epistemic_status == "abstain_enantiomer_underdetermined"

    # When C8 reflection-free visual encoder gives a strong margin (>0.05), resolve confidently!
    dec_vis = rerank_candidates(
        [(l_menthol, -0.30), (d_menthol, -2.10)],
        si.to_observables(),
        ablation="A6_full_posterior",
        n_conf=4,
        seed=42,
    )
    assert dec_vis.epistemic_status == "confident"
    assert dec_vis.canonical_smiles == l_menthol


def test_ablation_progression_and_targeted_epimer_rescue(tmp_path: Path):
    """Verify A0 -> A2 (HRMS) -> A3 (3D strain) -> A6 (NMR + epimer rescue) progression."""
    # cis-decalin-1-ol vs single-center C1 OCSR epimer misread & wrong-mass decoy
    truth = Chem.CanonSmiles("O[C@H]1CCC[C@@H]2CCCC[C@H]12")
    wrong_epimer = Chem.CanonSmiles("O[C@@H]1CCC[C@@H]2CCCC[C@H]12")
    wrong_mass = Chem.CanonSmiles("CC1CCC[C@@H]2CCCC[C@H]12")

    s_true = evaluate_3d_strain(truth, n_conf=6, seed=9)
    nmr_true = predict_nmr(truth, strain=s_true)
    mass = float(Descriptors.ExactMolWt(Chem.AddHs(Chem.MolFromSmiles(truth))))
    si, si_text = format_si_paragraph("decalol", formula="C10H18O", neutral_mass=mass, nmr=nmr_true)

    # Corrupt visual beam: wrong_mass is top-1 (-0.2), wrong_epimer is #2 (-0.5), truth MISSING!
    corrupted_beam = [(wrong_mass, -0.20), (wrong_epimer, -0.50)]

    d_a0 = rerank_candidates(corrupted_beam, si.to_observables(), ablation="A0_visual_top1")
    assert d_a0.canonical_smiles == wrong_mass

    d_a2 = rerank_candidates(corrupted_beam, si.to_observables(), ablation="A2_hrms", n_conf=4)
    assert d_a2.canonical_smiles == wrong_epimer  # HRMS eliminates wrong_mass!

    d_a6 = rerank_candidates(
        corrupted_beam, si.to_observables(), ablation="A6_full_posterior", n_conf=6, seed=9
    )
    assert d_a6.canonical_smiles == truth  # Epimer enumeration + 3D strain + NMR recovers truth!

    # Test CLI scripts/rerank_phys_spec.py across all ablations
    inp = tmp_path / "cands.jsonl"
    inp.write_text(
        json.dumps(
            {
                "id": "item_0",
                "candidates": [[wrong_mass, -0.2], [wrong_epimer, -0.5]],
                "si_text": si_text,
                "truth_smiles": truth,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "rerank_out"
    rc = rerank_cli_main(
        [
            "--input",
            str(inp),
            "--out",
            str(out_dir),
            "--ablation",
            "all",
            "--n-conf",
            "6",
            "--seed",
            "9",
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["A0_visual_top1"]["stereo_exact_match"]["mean"] == 0.0
    assert summary["A6_full_posterior"]["stereo_exact_match"]["mean"] == 1.0


def test_calibration_ece_and_aurc_metrics():
    confs = [0.95, 0.90, 0.85, 0.25, 0.15]
    good_acc = [True, True, True, False, False]
    bad_acc = [False, False, True, True, True]
    assert expected_calibration_error(confs, good_acc) < expected_calibration_error(confs, bad_acc)
    assert area_under_risk_coverage(confs, good_acc) < area_under_risk_coverage(confs, bad_acc)
