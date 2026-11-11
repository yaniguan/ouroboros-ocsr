"""Tests for offset-free MLIP committee force disagreement, LCMD diversity & scan CLI."""

from __future__ import annotations

import json
from pathlib import Path

from ouroboros.active.committee import evaluate_committee, select_active_batch
from ouroboros.active.motifs import build_bioisostere_panel
from ouroboros.active.oracle import build_pretrained_baseline_committee
from ouroboros.active.torsion import generate_blindspot_frames
from scripts.scan_mlip_blindspots import main as scan_cli_main


def test_offset_free_committee_uq_and_lcmd_scaffold_diversity():
    panel = build_bioisostere_panel("smoke")
    frames = []
    for rec in panel:
        frames.extend(generate_blindspot_frames(rec, torsion_step_deg=90.0, n_displaced=1, seed=0))

    committee = build_pretrained_baseline_committee()
    scores = evaluate_committee(frames, committee)

    # Planar/aliphatic controls have near-zero relative energy std DESPITE huge per-atom offsets!
    ctl_scores = [s for s in scores if s.group == "control"]
    bio_scores = [s for s in scores if s.group == "bioisostere"]
    assert max(s.rel_energy_std_kcal for s in ctl_scores) < 0.15
    assert (
        sum(s.sigma_f_max for s in bio_scores) / len(bio_scores)
        > 4.0 * (sum(s.sigma_f_max for s in ctl_scores) / len(ctl_scores))
    )

    # Diversity-aware qbc_lcmd_diverse visits strictly more scaffolds than greedy qbc_force!
    idx_greedy = select_active_batch(bio_scores, budget=10, strategy="qbc_force")
    idx_lcmd = select_active_batch(bio_scores, budget=10, strategy="qbc_lcmd_diverse")
    scaf_greedy = {bio_scores[i].mol_id for i in idx_greedy}
    scaf_lcmd = {bio_scores[i].mol_id for i in idx_lcmd}
    assert len(scaf_lcmd) > len(scaf_greedy)


def test_scan_mlip_blindspots_cli_quarantines_ineligible_elements(tmp_path: Path):
    lit_file = tmp_path / "ocsr_extracted.jsonl"
    rows = [
        {"id": "m1", "smiles": "O=C(NC)C12CC(C1)(C2)c1ccc(F)cc1"},  # BCP (eligible)
        {"id": "m2", "smiles": "O=C(NC)c1ccc([SF5])cc1"},  # SF5 alias (eligible)
        {"id": "m3", "smiles": "O=C(NC)c1ccc(C)cc1"},  # planar control (eligible)
        {"id": "m4", "smiles": "CO[Si](C)(C)c1ccccc1"},  # Si -> ineligible for MACE-OFF23!
        {"id": "m5", "smiles": "OB(O)c1ccccc1"},  # B -> ineligible for MACE-OFF23!
    ]
    lit_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / "scan_out"
    rc = scan_cli_main(
        [
            "--input-smiles",
            str(lit_file),
            "--torsion-step-deg",
            "90",
            "--n-displaced",
            "1",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "scan_summary.json").read_text(encoding="utf-8"))
    assert summary["n_eligible_scaffolds"] == 3
    assert summary["n_rejected_ineligible"] == 2
    assert summary["bioisostere_sigma_f_max"]["mean"] > summary["control_sigma_f_max"]["mean"]
