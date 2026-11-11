"""Tests for QM provenance guards, conservative E(3) autograd forces & H2.1-H2.3 distillation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ouroboros.active.distill import ConservativeShallowEnsembleDeltaMLIP
from ouroboros.active.oracle import QMLabel, verify_homogeneous_provenance
from scripts.run_active_distill import main as distill_cli_main

torch.set_num_threads(1)


def test_qm_provenance_guard_rejects_mixed_levels_of_theory():
    f_zero = np.zeros((3, 3), dtype=np.float64)
    l1 = QMLabel("f1", "m1", -100.0, f_zero, "psi4:wb97m-d3bj/def2-tzvppd/gas/q0/m1")
    l2 = QMLabel("f2", "m1", -12.0, f_zero, "tblite:gfn2-xtb/none/gas/q0/m1")
    assert verify_homogeneous_provenance([l1]) == l1.provenance_tag
    with pytest.raises(ValueError, match="incompatible QM provenance"):
        verify_homogeneous_provenance([l1, l2])


def test_shallow_ensemble_exact_conservative_forces_and_so3_equivariance():
    model = ConservativeShallowEnsembleDeltaMLIP(hidden_dim=16, n_rbf=12, n_heads=4, seed=7)
    z_idx = torch.tensor([1, 1, 2, 3, 4], dtype=torch.long)
    rng = np.random.default_rng(19)
    pos_np = rng.normal(scale=1.1, size=(5, 3))

    # 1. Verify exact conservative forces F_k = -dE_k/dR against central finite differences!
    pos_t = torch.tensor(pos_np, dtype=torch.float64, requires_grad=True)
    e_k, f_k = model.forward_energy_and_forces(z_idx, pos_t, create_graph=False)
    eps = 1e-6
    num_f0 = np.zeros_like(pos_np)
    for i in range(5):
        for d in range(3):
            p_plus = pos_np.copy()
            p_minus = pos_np.copy()
            p_plus[i, d] += eps
            p_minus[i, d] -= eps
            ep = model.forward_head_energies(z_idx, torch.tensor(p_plus, dtype=torch.float64))[0]
            em = model.forward_head_energies(z_idx, torch.tensor(p_minus, dtype=torch.float64))[0]
            num_f0[i, d] = -float((ep - em).item()) / (2.0 * eps)
    np.testing.assert_allclose(f_k[0].detach().numpy(), num_f0, atol=1e-6, rtol=1e-5)

    # 2. Verify E(3) energy invariance & O(3) force equivariance under rotation + reflection + shift
    q_mat, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # orthogonal O(3) matrix (det = +/- 1)
    shift = np.array([4.2, -3.1, 7.8])
    pos_rot = pos_np @ q_mat + shift
    e_rot, f_rot = model.forward_energy_and_forces(
        z_idx, torch.tensor(pos_rot, dtype=torch.float64, requires_grad=True)
    )
    np.testing.assert_allclose(e_rot.detach().numpy(), e_k.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(
        f_rot.detach().numpy(), f_k.detach().numpy() @ q_mat, atol=1e-9
    )


def test_end_to_end_h2_1_to_h2_3_active_distillation_cli(tmp_path: Path):
    out_dir = tmp_path / "distill_out"
    rc = distill_cli_main(
        [
            "--preset",
            "smoke",
            "--budgets",
            "8",
            "24",
            "--strategies",
            "random",
            "qbc_lcmd_diverse",
            "--torsion-step-deg",
            "60",
            "--n-displaced",
            "1",
            "--epochs",
            "80",
            "--seed",
            "3",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    rep = json.loads((out_dir / "distillation_report.json").read_text(encoding="utf-8"))

    # H2.1: Strained 3D bioisosteres show higher force disagreement & PES softening (<1.0)
    scan = rep["h2_1_blindspot_scan"]
    assert scan["bioisostere_sigma_f_max"]["mean"] > 4.0 * scan["control_sigma_f_max"]["mean"]
    assert scan["mann_whitney_p_value"] < 1e-4
    base_bio_mae = scan["baseline_test_bioisostere_rel_mae_kcal"]["mean"]
    base_alpha = scan["baseline_test_bioisostere_softening_alpha"]["mean"]
    assert base_bio_mae > 1.0
    assert base_alpha < 0.70

    # H2.2 & H2.3: Active distillation recovers <0.5 kcal/mol accuracy, lifts alpha toward 1.0,
    # outperforms random acquisition at low budget, and avoids catastrophic control forgetting!
    lc_lcmd_8 = rep["learning_curves"]["qbc_lcmd_diverse"]["8"]
    lc_rand_8 = rep["learning_curves"]["random"]["8"]
    lc_lcmd_24 = rep["learning_curves"]["qbc_lcmd_diverse"]["24"]

    assert (
        lc_lcmd_8["test_bioisostere_force_rmse_kcal_a"]["mean"]
        <= lc_rand_8["test_bioisostere_force_rmse_kcal_a"]["mean"]
    )
    assert lc_lcmd_24["test_bioisostere_rel_mae_kcal"]["mean"] < 0.50
    assert lc_lcmd_24["test_bioisostere_rel_mae_kcal"]["mean"] < 0.35 * base_bio_mae
    assert lc_lcmd_24["test_bioisostere_softening_alpha"]["mean"] > 0.82
    assert lc_lcmd_24["test_control_forgetting_drift_kcal"]["mean"] < 0.12


def test_mmff_cache_and_distilled_ase_calculator_nve_conservation():
    from ase import Atoms, units
    from ase.md.verlet import VelocityVerlet

    from ouroboros.active.distill import (
        DistilledASECalculator,
        DistilledEnsemblePotential,
    )
    from ouroboros.active.motifs import build_bioisostere_panel
    from ouroboros.active.oracle import (
        AnalyticalCageMember,
        CachedPredictorMember,
        MMFF94sMember,
    )
    from ouroboros.active.torsion import generate_blindspot_frames

    rec = build_bioisostere_panel("smoke")[0]
    fr = generate_blindspot_frames(rec, torsion_step_deg=90.0, n_displaced=0)[0]

    mmff = MMFF94sMember()
    cached = CachedPredictorMember(mmff)
    e1, f1 = cached.predict_frame(fr)
    e2, f2 = cached.predict_frame(fr)
    assert np.isfinite(e1) and f1.shape == fr.positions.shape
    assert e1 == e2 and np.allclose(f1, f2)
    assert len(cached.cache) == 1

    base = AnalyticalCageMember(0.5, 0.5)
    delta = ConservativeShallowEnsembleDeltaMLIP(hidden_dim=12, n_rbf=10, n_heads=4, seed=2)
    pot = DistilledEnsemblePotential(base, delta)
    calc = DistilledASECalculator(pot, mol_id=rec.mol_id, smiles=rec.smiles)

    atoms = Atoms(numbers=fr.atomic_numbers.tolist(), positions=fr.positions.copy())
    atoms.calc = calc
    rng = np.random.default_rng(11)
    atoms.set_momenta(rng.normal(scale=0.02, size=fr.positions.shape))
    e_tot_0 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    dyn = VelocityVerlet(atoms, timestep=0.25 * units.fs)
    dyn.run(6)
    e_tot_1 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    assert abs(e_tot_1 - e_tot_0) < 1e-3
    assert calc.n_evals >= 6
    assert calc.last_sigma_f_max >= 0.0
