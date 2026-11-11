"""Unit tests for MACE-OFF23 10-element eligibility, 7 bioisostere detectors & torsion softening."""

from __future__ import annotations

import numpy as np
import pytest

from ouroboros.active.motifs import (
    AIMNET2_POLICY,
    BIOISOSTERE_FAMILIES,
    build_bioisostere_panel,
    check_mace_off23_eligibility,
    detect_bioisostere_families,
)
from ouroboros.active.torsion import compute_softening_diagnostics, generate_blindspot_frames


def test_mace_off23_vs_aimnet2_eligibility_gates():
    # Neutral C/H/N/O/F/S bioisosteres (with bracket aliases) pass MACE_OFF23_POLICY
    ok_sf5, canon_sf5, reasons_sf5 = check_mace_off23_eligibility("O=C(NC)c1ccc([SF5])cc1")
    assert ok_sf5 and canon_sf5 is not None and reasons_sf5 == ()

    # Boron, silicon, net charge, and radicals MUST be rejected by MACE_OFF23_POLICY
    ok_b, _, r_b = check_mace_off23_eligibility("Cc1ccc(B(O)O)cc1")
    assert not ok_b and any("unsupported_elements:B" in r for r in r_b)

    ok_si, _, r_si = check_mace_off23_eligibility("CO[Si](C)(C)C")
    assert not ok_si and any("unsupported_elements:Si" in r for r in r_si)

    ok_q, _, r_q = check_mace_off23_eligibility("C[N+](C)(C)CC(=O)O")
    assert not ok_q and "residual_charge" in r_q

    ok_rad, _, r_rad = check_mace_off23_eligibility("[CH2]c1ccccc1")
    assert not ok_rad and "radicals" in r_rad

    # AIMNet2 (14 elements, allow_charge=True) accepts B, Si, and charged ions
    ok_aim_b, _, _ = check_mace_off23_eligibility("Cc1ccc(B(O)O)cc1", policy=AIMNET2_POLICY)
    ok_aim_q, _, _ = check_mace_off23_eligibility("C[N+](C)(C)CC(=O)O", policy=AIMNET2_POLICY)
    assert ok_aim_b and ok_aim_q


def test_all_seven_post2020_bioisostere_families_detected_and_full_panel_unique():
    exemplars = {
        "bcp_111": "O=C(NC)C12CC(C1)(C2)c1ccc(F)cc1",
        "bch_211": "O=C(OC)C12CCC(C1)(C#N)C2",
        "oxa_bch_211": "O=C(NC)C12COC(C1)(C)C2",
        "bchep_311": "O=C(OC)C12CCCC(C1)(F)C2",
        "cubane": "O=C(NC)C12C3C4C1C5C2C3C45C",
        "azaspiro_oxetane": "O=C(C)N1CC2(C1)COC2",
        "sf5_polyfluoro": "O=C(NC)c1ccc(S(F)(F)(F)(F)F)cc1",
    }
    for fam, smi in exemplars.items():
        detected = detect_bioisostere_families(smi)
        assert fam in detected, f"expected {fam} in {detected} for {smi}"

    assert detect_bioisostere_families("O=C(NC)c1ccc(C)cc1") == ()

    panel = build_bioisostere_panel("full")
    assert len(panel) == 112  # 56 bioisosteres + 56 matched controls
    assert len({r.connectivity_key for r in panel}) == 112
    present_fams = {r.family for r in panel if r.group == "bioisostere"}
    assert present_fams == set(BIOISOSTERE_FAMILIES)


def test_bcp_torsion_scan_transannular_distance_and_softening_slope():
    panel = build_bioisostere_panel("smoke")
    bcp_rec = next(r for r in panel if r.family == "bcp_111")
    frames = generate_blindspot_frames(bcp_rec, torsion_step_deg=60.0, n_displaced=2, seed=3)
    assert len(frames) >= 7
    # BCP nonbonded bridgehead C1...C3 distance sits in the strained ~1.80-1.98 A window!
    for fr in frames:
        assert fr.transannular_c1_c3_a is not None
        assert 1.70 <= fr.transannular_c1_c3_a <= 2.05

    ref_e = [0.0, 1.5, 4.0, 2.0, 0.5, 3.5, 1.0][: len(frames)]
    soft_pred_e = [0.52 * x + 100.0 for x in ref_e]  # constant offset + 0.52 softening slope!
    zero_f = [np.zeros_like(fr.positions) for fr in frames]
    diag = compute_softening_diagnostics(frames, soft_pred_e, zero_f, ref_e, zero_f)
    assert diag.softening_slope_alpha == pytest.approx(0.52, abs=1e-6)
    assert diag.barrier_height_error_kcal == pytest.approx(0.48 * max(ref_e), abs=1e-6)
