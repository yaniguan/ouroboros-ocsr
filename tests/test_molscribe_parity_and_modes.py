"""Tests for MolScribe scoring parity, harness --protocol flag, edge_plan runners,
and committed operating-mode / rotated-real benchmark receipts (#134)."""

from __future__ import annotations

import json
from pathlib import Path

from lab import edge_plan as ep
from ouroboros.eval import harness
from ouroboros.eval.metrics import score_pair
from ouroboros.eval.molscribe_parity import (
    aggregate_molscribe,
    score_molscribe_pair,
    standardize_molscribe,
)

ROOT = Path(__file__).resolve().parents[1]


def test_molscribe_standardization_keeps_tetrahedral_strips_ez_and_normalizes_rgroups():
    # 1. E vs Z alkenes differ under strict Ouroboros scoring, but match under MolScribe protocol
    assert score_pair("C/C=C/C", "C/C=C\\C").exact is False
    assert score_molscribe_pair("C/C=C/C", "C/C=C\\C").exact is True

    # 2. Tetrahedral (R) vs (S) enantiomers remain distinct under BOTH protocols
    assert score_pair("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O").exact is False
    assert score_molscribe_pair("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O").exact is False
    assert score_molscribe_pair("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O").exact_flat is True

    # 3. Monovalent abbreviations expand ([OMe] -> OC) and generic R-groups collapse ([R1] -> *)
    ome_smi = standardize_molscribe("c1ccc([OMe])cc1").smiles
    assert ome_smi == standardize_molscribe("COc1ccccc1").smiles
    assert standardize_molscribe("[R1]c1ccccc1").smiles == standardize_molscribe("*c1ccccc1").smiles

    agg = aggregate_molscribe(
        [
            score_molscribe_pair("C/C=C/C", "C/C=C\\C"),
            score_molscribe_pair("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"),
        ],
        n_boot=100,
    )
    assert agg["n"] == 2
    assert agg["exact"] == 0.5
    assert agg["exact_flat"] == 1.0


def test_harness_protocol_both_and_frozen_test_v2_parity_receipt(tmp_path):
    preds_csv = ROOT / "benchmarks" / "frozen-test-v2" / "AC-ensemble-beam5-ttapad-preds.csv"
    out_json = tmp_path / "both.json"
    out_md = tmp_path / "both.md"
    rc = harness.main(
        [
            str(preds_csv),
            "--protocol",
            "both",
            "--n-boot",
            "100",
            "--json",
            str(out_json),
            "--md",
            str(out_md),
            "--quiet",
        ]
    )
    assert rc == 0
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["protocol"] == "both"
    ouro_all = next(g for g in rep["groups"] if g["subset"] == "ALL")
    ms_all = next(g for g in rep["molscribe_groups"] if g["subset"] == "ALL")
    assert round(ouro_all["exact"], 4) == 0.4667
    assert ms_all["exact"] >= ouro_all["exact"]
    assert "## MolScribe Protocol Parity" in out_md.read_text(encoding="utf-8")

    committed = json.loads(
        (ROOT / "benchmarks" / "frozen-test-v2" / "molscribe_parity_report.json").read_text(
            encoding="utf-8"
        )
    )
    committed_ms_all = next(g for g in committed["molscribe_groups"] if g["subset"] == "ALL")
    assert round(committed_ms_all["exact"], 4) == round(ms_all["exact"], 4)


def test_edge_plan_emit_shell_and_run_step(tmp_path, monkeypatch):
    sh_path = tmp_path / "distill.sh"
    assert ep.main(["--emit-shell", str(sh_path)]) == 0
    script = sh_path.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    assert "scripts/predict_real.py" in script

    ran = []

    class _DummyProc:
        returncode = 0

    def _fake_run(cmd, *, shell, check):
        ran.append((cmd, shell))
        return _DummyProc()

    monkeypatch.setattr(ep.subprocess, "run", _fake_run)
    assert ep.main(["--run-step", "4"]) == 0
    assert len(ran) == 1 and "check_leakage.py" in ran[0][0]
    assert ep.main(["--run-step", "99"]) == 1


def test_committed_operating_modes_and_rotated_real_receipts():
    modes_file = ROOT / "benchmarks" / "cpu-api" / "operating_modes.json"
    assert modes_file.is_file()
    modes = json.loads(modes_file.read_text(encoding="utf-8"))
    assert set(modes["modes"]) == {
        "A_int8_greedy",
        "C_pure_int8_greedy",
        "AC_int8_beam5",
        "AC_int8_beam5_ttapad_gated",
    }
    assert (
        modes["modes"]["AC_int8_beam5_ttapad_gated"]["wall_ms"]["p50"]
        > modes["modes"]["A_int8_greedy"]["wall_ms"]["p50"]
    )

    rot_file = ROOT / "benchmarks" / "rotated-real-uob" / "summary.json"
    assert rot_file.is_file()
    rot = json.loads(rot_file.read_text(encoding="utf-8"))
    assert rot["n_images"] == 100
    assert set(rot["arms"]) == {"A", "B", "C_pure"}
    # On rotated 90/180/270 real UOB figures, C_pure (zero rotated training images) strongly beats A
    assert (
        rot["arms"]["C_pure"]["rotated_90_180_270_mean_exact"]
        > rot["arms"]["A"]["rotated_90_180_270_mean_exact"] + 0.15
    )


def test_browser_lab_uses_dimensionless_strain_units_on_heuristic_custom_smiles():
    lab_js = (ROOT / "site" / "lab" / "lab.js").read_text(encoding="utf-8")
    assert "Spring-VSEPR geometric strain index" in lab_js
    assert "Strain (arb. u.)" in lab_js
