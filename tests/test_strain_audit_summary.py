import json
from pathlib import Path

import pytest

from ouroboros.phys.strain import summarize_strain_audit

RECEIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "p1_chemveritas" / "mace_strain_audit.json"


def _pair(delta, ok=True):
    return {"truth_embed_ok": True, "corrupt_embed_ok": ok,
            "delta_e_mace_kcal": delta, "delta_e_mmff_kcal": delta}


def test_embed_failures_are_excluded_from_strain_stats():
    s = summarize_strain_audit({"pairs": [_pair(1.0), _pair(3.0), _pair(-1.0), _pair(45.0, ok=False)]})
    assert s["n_pairs"] == 4
    assert s["n_embed_failures"] == 1
    assert s["n_measured_pairs"] == 3
    assert s["delta_e_mace_kcal"]["median"] == 1.0
    assert s["delta_e_mace_kcal"]["max"] == 3.0
    assert s["delta_e_mace_kcal"]["n_corrupt_higher"] == 2


def test_all_failures_give_no_stats():
    s = summarize_strain_audit({"pairs": [_pair(45.0, ok=False)]})
    assert s["n_measured_pairs"] == 0
    assert s["delta_e_mace_kcal"] is None


@pytest.mark.skipif(not RECEIPT.exists(), reason="strain receipt not present")
def test_committed_receipt_summary():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    s = summarize_strain_audit(receipt)
    assert s["n_pairs"] == receipt["n_pairs"] == 50
    assert s["n_embed_failures"] == receipt["n_embed_or_cip_failures_on_corrupt"] == 30
    assert s["n_measured_pairs"] == 20
    assert s["delta_e_mace_kcal"]["max"] < 45.0


@pytest.mark.skipif(not RECEIPT.exists(), reason="strain receipt not present")
def test_committed_summary_is_current():
    summary = RECEIPT.with_name("mace_strain_audit_summary.json")
    expected = summarize_strain_audit(json.loads(RECEIPT.read_text(encoding="utf-8")))
    assert json.loads(summary.read_text(encoding="utf-8")) == expected


def test_penalty_in_measured_pair_is_an_error():
    with pytest.raises(ValueError):
        summarize_strain_audit({"pairs": [_pair(45.0)]})
