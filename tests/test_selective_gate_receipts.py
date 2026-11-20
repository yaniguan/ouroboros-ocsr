"""Verify validation-locked confidence gate limits and single-shot frozen-test-v2 receipt."""

from __future__ import annotations

import json
from pathlib import Path

from lab import edge_plan as ep

ROOT = Path(__file__).resolve().parents[1]


def test_selective_gate_limits_locked_on_pmc_val():
    cfg = ep.validate(ep.load_config(ep.DEFAULT_CONFIG))
    pl = cfg["pseudo_labels"]
    assert pl["min_margin"] == 0.05
    assert pl["max_disagreement"] == 0.0

    bench = ROOT / "benchmarks/selective-gate"
    val_dis0 = json.loads((bench / "pmc_val_disagreement_0.0.json").read_text(encoding="utf-8"))
    by_margin = {r["min_margin"]: r["overall"] for r in val_dis0}
    chosen = by_margin[0.05]
    assert chosen["n"] == 963
    assert chosen["accepted"] == 192
    assert chosen["precision"] > 0.90

    frozen = json.loads((bench / "frozen_test_v2_gated_summary.json").read_text(encoding="utf-8"))
    assert frozen["eval_set"] == "frozen-test-v2"
    assert frozen["locked_on"] == "pmc_val"
    assert frozen["min_margin"] == pl["min_margin"]
    assert frozen["max_disagreement"] == pl["max_disagreement"]
    assert frozen["overall"]["n"] == 300
    assert 0 < frozen["overall"]["accepted"] <= 300
    assert frozen["overall"]["precision"] > frozen["overall"]["base_exact"]
