"""End-to-end tests for ChemVeritas-Bench (Gold-600 catalog, 4-view builder, RPF & CLI harness)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ouroboros.bench.catalog import STRATA, build_catalog
from ouroboros.bench.vlm_harness import parse_vlm_response
from scripts.build_phys_bench import main as build_bench_main
from scripts.eval_chemveritas import main as eval_bench_main

torch.set_num_threads(1)


def test_gold600_catalog_has_600_unique_connectivity_scaffolds_and_exclusion_gate():
    entries = build_catalog(preset="gold600")
    assert len(entries) == 600
    assert len({e.inchikey14 for e in entries}) == 600
    for stratum in STRATA:
        assert sum(1 for e in entries if e.stratum == stratum) == 120

    # Verify connectivity-block zero-leakage exclusion filter
    blocked = {entries[0].inchikey14, entries[125].inchikey14}
    filtered = build_catalog(preset="smoke", exclude_key14=blocked)
    assert all(e.inchikey14 not in blocked for e in filtered)


def test_end_to_end_chemveritas_smoke_build_and_evaluation(tmp_path: Path):
    bench_dir = tmp_path / "chemveritas_smoke"
    rc = build_bench_main(
        [
            "--out",
            str(bench_dir),
            "--preset",
            "smoke",
            "--image-size",
            "192",
            "--n-conf",
            "2",
            "--seed",
            "2026",
        ]
    )
    assert rc == 0
    manifest = json.loads((bench_dir / "manifest_receipt.json").read_text(encoding="utf-8"))
    assert manifest["n_molecules"] == 25
    assert manifest["n_t1_visual_instances"] == 100  # 25 molecules * 4 views
    assert manifest["unique_inchikey14"] == 25

    # Verify physics_ablation mode: A6_full_posterior outperforms A0_visual_top1 on T2 & T5
    eval_a0 = tmp_path / "eval_a0"
    eval_a6 = tmp_path / "eval_a6"
    assert (
        eval_bench_main(
            [
                "--bench-dir",
                str(bench_dir),
                "--task",
                "all",
                "--mode",
                "physics_ablation",
                "--ablation",
                "A0_visual_top1",
                "--n-conf",
                "2",
                "--out",
                str(eval_a0),
            ]
        )
        == 0
    )
    assert (
        eval_bench_main(
            [
                "--bench-dir",
                str(bench_dir),
                "--task",
                "all",
                "--mode",
                "physics_ablation",
                "--ablation",
                "A6_full_posterior",
                "--n-conf",
                "2",
                "--out",
                str(eval_a6),
            ]
        )
        == 0
    )

    rep_a0 = json.loads((eval_a0 / "eval_summary.json").read_text(encoding="utf-8"))["ablations"][
        "A0_visual_top1"
    ]
    rep_a6 = json.loads((eval_a6 / "eval_summary.json").read_text(encoding="utf-8"))["ablations"][
        "A6_full_posterior"
    ]
    assert rep_a6["T2"]["stereo_exact_match"]["mean"] > rep_a0["T2"]["stereo_exact_match"]["mean"]
    assert rep_a6["T2"]["veritas_rpf"]["mean"] >= 0.80
    # T5: A6 abstains 100% on achiral-only enantiomer pairs; naive A0 hallucinates chirality!
    assert rep_a6["T5"]["epistemic_abstention_rate"]["mean"] == 1.0
    assert rep_a6["T5"]["hallucinated_chirality_rate"] == 0.0
    assert rep_a0["T5"]["epistemic_abstention_rate"]["mean"] == 0.0

    # Verify T1 external OCSR import keeps V0_clean and V1-V3 strictly separate with bootstrap CIs
    t1_text = (bench_dir / "task_T1_visual_ocsr.jsonl").read_text(encoding="utf-8")
    t1_rows = [json.loads(line) for line in t1_text.splitlines()]
    pred_file = tmp_path / "mock_ocsr.jsonl"
    pred_lines = []
    for row in t1_rows:
        # Simulate clean V0 succeeding and crowded scan V2 suffering stereochemical corruption
        smi = row["truth_smiles"] if row["view"] == "V0_clean" else "CC(=O)O"
        pred_lines.append(json.dumps({"task_id": row["task_id"], "smiles": smi, "confidence": 0.9}))
    pred_file.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

    imp_out = tmp_path / "eval_import"
    assert (
        eval_bench_main(
            [
                "--bench-dir",
                str(bench_dir),
                "--task",
                "T1",
                "--mode",
                "import_preds",
                "--predictions",
                str(pred_file),
                "--out",
                str(imp_out),
            ]
        )
        == 0
    )
    t1_rep = json.loads((imp_out / "eval_summary.json").read_text(encoding="utf-8"))["tasks"]["T1"]
    assert set(t1_rep.keys()) == {
        "V0_clean",
        "V1_rotated_offgrid",
        "V2_crowded_si_scan",
        "V3_condensed_layout",
    }
    assert t1_rep["V0_clean"]["stereo_exact_match"]["mean"] == 1.0
    assert t1_rep["V2_crowded_si_scan"]["stereo_exact_match"]["mean"] == 0.0
    assert t1_rep["V0_clean"]["stereo_exact_match"]["n"] == 25

    # Verify prompt export & VLM fenced-JSON parser
    exp_out = tmp_path / "prompts"
    assert (
        eval_bench_main(
            [
                "--bench-dir",
                str(bench_dir),
                "--task",
                "T2",
                "--mode",
                "export_prompts",
                "--out",
                str(exp_out),
            ]
        )
        == 0
    )
    parsed = parse_vlm_response('Reasoning...\n```json\n{"smiles": "CCO", "confidence": 0.88}\n```')
    assert parsed["smiles"] == "CCO" and parsed["confidence"] == 0.88


def test_tier2_lit_nmr_panel_monotonic_disambiguation():
    from ouroboros.bench.lit_nmr_panel import evaluate_lit_nmr_panel, get_lit_nmr_panel

    panel = get_lit_nmr_panel()
    assert len(panel) == 30
    assert len({e.pair_id for e in panel}) == 15
    assert all(e.doi for e in panel)

    report = evaluate_lit_nmr_panel(n_conf=8, calc=None)
    abls = report["ablations"]
    a0 = abls["A0_visual_top1"]["exact_stereoisomer_accuracy"]["mean"]
    a3 = abls["A3_strain_3d"]["exact_stereoisomer_accuracy"]["mean"]
    a4 = abls["A4_nmr_13c"]["exact_stereoisomer_accuracy"]["mean"]
    a5 = abls["A5_nmr_13c_1h_j"]["exact_stereoisomer_accuracy"]["mean"]
    a6 = abls["A6_full_posterior"]["exact_stereoisomer_accuracy"]["mean"]
    assert a0 == 0.0
    assert a3 > a0
    assert a4 > a3
    assert a5 >= a4
    assert a6 >= 0.70


def test_strain_graceful_fallback_on_silyl_and_hypervalent_candidates() -> None:
    from ouroboros.phys.strain import evaluate_3d_strain

    class _DummyMaceCalc:
        _cache_tag = "mace_dummy"
        _relax_steps = 0

        def __getattr__(self, name: str):
            raise RuntimeError("MACE should not be called on Si-containing molecules")

    # Silyl ether (Z=14 outside MACE-OFF23 10-element set) must fall back cleanly to MMFF/UFF
    sr_si = evaluate_3d_strain(
        "CC(C)(C)[Si](C)(C)O[C@H]1CCCC[C@@H]1C", n_conf=2, calc=_DummyMaceCalc()
    )
    assert sr_si.valid and sr_si.embed_ok and sr_si.min_energy_kcal is not None

    # Hypervalent -SF5 (triggers UFF fallback without crashing AngleBend.cpp)
    sr_sf5 = evaluate_3d_strain("FS(F)(F)(F)(F)c1ccc(O)cc1", n_conf=2, calc=None)
    assert sr_sf5.valid and sr_sf5.embed_ok

