"""Unit tests for HRMS exact-mass matching, superatom expansion, and 3D Boltzmann NMR inversion."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from ouroboros.eval.standardize import standardize
from ouroboros.geometry.propagation import categorize
from ouroboros.phys.epimers import (
    condense_superatom_for_depiction,
    enumerate_targeted_epimers,
    expand_superatoms,
)
from ouroboros.phys.hrms import exact_adduct_mz, score_hrms
from ouroboros.phys.nmr import optimal_1d_match_mae, predict_nmr, score_nmr


def test_hrms_exact_adduct_and_ppm_window():
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin, C9H8O4
    mz_na = exact_adduct_mz(smi, "[M+Na]+")
    assert mz_na is not None

    # 1.8 ppm perturbation stays strictly inside +/- 5 ppm window with zero loss
    obs_ok = mz_na * (1.0 + 1.8e-6)
    res_ok = score_hrms(smi, obs_ok, adduct="[M+Na]+", expected_formula="C9H8O4")
    assert res_ok.within_window
    assert res_ok.loss == 0.0
    assert res_ok.ppm_error is not None and abs(res_ok.ppm_error - 1.8) < 0.05

    # Near-mass decoy (+CH2) fails HRMS window with large smooth loss
    decoy = "CCC(=O)Oc1ccccc1C(=O)O"
    res_bad = score_hrms(decoy, obs_ok, adduct="[M+Na]+")
    assert not res_bad.within_window
    assert res_bad.loss > 20.0

    # Same-formula constitutional isomer has 0 ppm HRMS error (proving HRMS needs NMR!)
    regio = "COC(=O)c1ccccc1OC=O"
    res_iso = score_hrms(regio, mz_na, adduct="[M+Na]+")
    assert res_iso.within_window
    assert res_iso.ppm_error is not None and res_iso.ppm_error < 0.01


def test_superatom_expansion_preserves_chirality_and_handles_common_groups():
    r_boc = expand_superatoms("C[C@H]([Boc])N")
    s_boc = expand_superatoms("C[C@@H]([Boc])N")
    st_r = standardize(r_boc)
    st_s = standardize(s_boc)
    assert st_r.valid and st_s.valid
    assert categorize(st_r.smiles, st_s.smiles) == "enantiomer"

    for raw in (
        "c1ccc([SF5])cc1",
        "OCC([TBS])C",
        "N([Fmoc])CC(=O)O",
        "Cc1ccc([Bpin])cc1",
        "C([Cbz])N1CCCC1",
    ):
        expanded = expand_superatoms(raw)
        st = standardize(expanded)
        assert st.valid, f"failed to expand {raw} -> {expanded}"

    full = "CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)O"
    cond, changed = condense_superatom_for_depiction(full)
    assert changed and "[Boc]" in cond
    roundtrip = standardize(expand_superatoms(cond))
    assert roundtrip.smiles == standardize(full).smiles


def test_meso_vs_unsymmetric_c13_symmetry_signal_count():
    # meso-tartaric acid has an improper mirror plane (achiral) -> 2 unique 13C signals
    meso = "O=C(O)[C@H](O)[C@@H](O)C(=O)O"
    chiral_rr = "O=C(O)[C@H](O)[C@H](O)C(=O)O"
    pred_meso = predict_nmr(meso, n_conf=4, seed=7)
    pred_rr = predict_nmr(chiral_rr, n_conf=4, seed=7)
    assert pred_meso.c13_sym_count == 2
    assert pred_rr.c13_sym_count == 2

    # Unsymmetric 5-carbon diol has 5 distinct 13C signals
    unsym = "C[C@H](O)[C@@H](O)CC"
    pred_unsym = predict_nmr(unsym, n_conf=4, seed=7)
    assert pred_unsym.c13_sym_count == 5


def test_diastereomer_3d_boltzmann_nmr_and_karplus_discrimination():
    # cis- vs trans-1,2-dimethylcyclohexane: axial methyls induce 3D gamma-gauche upfield shifts
    from ouroboros.phys.strain import evaluate_3d_strain

    trans_12 = "C[C@H]1CCCC[C@@H]1C"
    cis_12 = "C[C@H]1CCCC[C@H]1C"
    s_trans = evaluate_3d_strain(trans_12, n_conf=6, seed=11)
    s_cis = evaluate_3d_strain(cis_12, n_conf=6, seed=11)
    p_trans = predict_nmr(trans_12, strain=s_trans)
    p_cis = predict_nmr(cis_12, strain=s_cis)

    # Self-match beats cross-diastereomer match
    self_match = score_nmr(trans_12, obs_c13_ppm=p_trans.c13_shifts_ppm, strain=s_trans)
    cross_match = score_nmr(cis_12, obs_c13_ppm=p_trans.c13_shifts_ppm, strain=s_cis)
    assert self_match.c13_mae_ppm is not None and cross_match.c13_mae_ppm is not None
    assert self_match.c13_mae_ppm < 1e-6
    assert cross_match.c13_mae_ppm > 0.25 or p_trans.vicinal_j_hh_hz != p_cis.vicinal_j_hh_hz


def test_optimal_monotone_dp_peak_matching():
    pred = [170.2, 129.5, 128.1, 21.0]
    obs = [170.0, 129.4, 128.3, 55.0, 21.1]  # extra impurity/solvent peak at 55.0 skipped!
    mae = optimal_1d_match_mae(pred, obs)
    assert mae == pytest.approx((0.2 + 0.1 + 0.2 + 0.1) / 4.0, abs=1e-6)


def test_targeted_epimers_stay_same_connectivity():
    base = "CC1=CC[C@H]([C@@H](C)O)CC1"
    eps = enumerate_targeted_epimers([base], max_centers=2, max_epimers=6)
    assert len(eps) >= 2
    w0 = Descriptors.ExactMolWt(Chem.AddHs(Chem.MolFromSmiles(base)))
    for ep in eps:
        w = Descriptors.ExactMolWt(Chem.AddHs(Chem.MolFromSmiles(ep)))
        assert abs(w - w0) < 1e-9
