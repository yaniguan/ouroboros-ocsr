"""Unit tests for Supporting Information analytical parser & CLI extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouroboros.si.parser import parse_si_block, parse_si_document
from scripts.extract_si_bench import main as extract_si_main


def test_parse_si_block_strips_cdcl3_solvent_and_parses_j_couplings():
    paragraph = (
        "Compound 14b: [α]D +42.5 (c 1.0, CHCl3); Molecular Formula: C15H18O3. "
        "1H NMR (500 MHz, CDCl3) δ 7.28 (d, J = 8.4, 2.1 Hz, 2H), 4.12 (q, J = 7.1 Hz, 2H), "
        "1.35 (s, 3H). "
        "13C NMR (126 MHz, CDCl3) δ 172.4, 158.2, 130.1, 114.3, 77.16, 60.8, 24.5. "
        "HRMS (ESI-TOF) m/z: [M + Na]+ calcd for C15H18O3Na 269.1148; found 269.1152."
    )
    rec = parse_si_block(paragraph, compound_id="14b")
    assert rec.compound_id == "14b"
    assert rec.molecular_formula == "C15H18O3"
    assert rec.hrms_adduct == "[M+Na]+"
    assert rec.hrms_calcd_mz == pytest.approx(269.1148)
    assert rec.hrms_obs_mz == pytest.approx(269.1152)
    assert rec.internal_hrms_ppm is not None and rec.internal_hrms_ppm < 2.0
    # 77.16 CDCl3 residual peak stripped automatically!
    assert 77.16 not in rec.c13_peaks_ppm
    assert len(rec.c13_peaks_ppm) == 6
    assert len(rec.h1_multiplets) == 3
    assert rec.h1_multiplets[0].j_hz == (8.4, 2.1)
    assert rec.optical_rotation_deg == pytest.approx(42.5)


def test_parse_si_document_and_extract_cli_filter(tmp_path: Path):
    si_doc = (
        "Compound S1: Molecular Formula: C10H12O2. "
        "13C NMR (101 MHz, CDCl3) δ 170.5, 136.2, 128.5, 128.1, 66.4, 21.0. "
        "HRMS (ESI) m/z: [M+H]+ calcd for 165.0910; found 165.0913.\n\n"
        "Compound S2: Molecular Formula: C10H12O2. "
        "13C NMR (101 MHz, DMSO-d6) δ 170.1, 136.0, 39.52, 20.8. "
        "HRMS (ESI) m/z: [M+H]+ calcd for 165.0910; found 165.1050.\n"
    )
    parsed = parse_si_document(si_doc)
    assert len(parsed) == 2
    assert 39.52 not in parsed[1].c13_peaks_ppm  # DMSO-d6 solvent peak stripped!

    si_file = tmp_path / "paper_si.txt"
    si_file.write_text(si_doc, encoding="utf-8")
    out_jsonl = tmp_path / "filtered_si.jsonl"

    rc = extract_si_main(
        ["--si-text", str(si_file), "--out", str(out_jsonl), "--max-internal-ppm", "5.0"]
    )
    assert rc == 0
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    # S1 (~1.8 ppm) passes; S2 (~84.8 ppm typo) is rejected by the 5 ppm consistency gate!
    assert len(rows) == 1
    assert rows[0]["compound_id"] == "S1"
