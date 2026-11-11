"""Evaluate physics ablations, imported OCSR predictions, or frontier VLMs on ChemVeritas-Bench.

Supports all 5 ChemVeritas tasks (T1-T5) while keeping rendered (V0) and rotated/degraded
(V1-V3) views strictly segregated with sample counts and bootstrap 95% confidence intervals.

Examples:
    python scripts/eval_chemveritas.py --bench-dir results/chemveritas_smoke --mode physics_ablation
    python scripts/eval_chemveritas.py --bench-dir results/chemveritas_smoke --mode export_prompts \
        --out results/vlm_prompts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ouroboros.bench.scoring import (
    evaluate_t1_visual_predictions,
    evaluate_t2_multimodal_predictions,
    evaluate_t3_audit_predictions,
    evaluate_t4_thermo_predictions,
    evaluate_t5_parity_abstention,
)
from ouroboros.bench.vlm_harness import (
    build_vlm_prompt,
    call_frontier_vlm,
    import_external_ocsr_predictions,
    run_physics_posterior_baseline,
)
from ouroboros.phys.posterior import ABLATION_PRESETS

TASK_FILES = {
    "T1": "task_T1_visual_ocsr.jsonl",
    "T2": "task_T2_multimodal_si_disambig.jsonl",
    "T3": "task_T3_silent_error_audit.jsonl",
    "T4": "task_T4_thermodynamic_ranking.jsonl",
    "T5": "task_T5_epistemic_parity_abstention.jsonl",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _score_task(task_key: str, items: list[dict], preds: dict[str, dict], seed: int) -> dict:
    if task_key == "T1":
        return evaluate_t1_visual_predictions(items, preds, compute_3d_energy_error=True, seed=seed)
    if task_key == "T2":
        return evaluate_t2_multimodal_predictions(items, preds, seed=seed)
    if task_key == "T3":
        return evaluate_t3_audit_predictions(items, preds, seed=seed)
    if task_key == "T4":
        return evaluate_t4_thermo_predictions(items, preds, seed=seed)
    if task_key == "T5":
        return evaluate_t5_parity_abstention(items, preds, seed=seed)
    raise ValueError(f"unknown task {task_key!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-dir", type=Path, required=True, help="ChemVeritas benchmark directory")
    p.add_argument(
        "--task",
        choices=("T1", "T2", "T3", "T4", "T5", "all"),
        default="all",
        help="Task subset to evaluate",
    )
    p.add_argument(
        "--mode",
        choices=("physics_ablation", "import_preds", "export_prompts", "live_vlm"),
        default="physics_ablation",
    )
    p.add_argument(
        "--ablation",
        choices=(*ABLATION_PRESETS.keys(), "all"),
        default="A6_full_posterior",
        help="Ablation preset for physics_ablation mode",
    )
    p.add_argument("--predictions", type=Path, default=None, help="CSV/JSONL for import_preds mode")
    p.add_argument("--provider", choices=("anthropic", "openai", "gemini"), default="gemini")
    p.add_argument("--model", type=str, default="gemini-2.5-pro")
    p.add_argument("--n-conf", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output directory for evaluation JSON")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = list(TASK_FILES) if args.task == "all" else [args.task]
    report: dict[str, object] = {"mode": args.mode}

    if args.mode == "export_prompts":
        exported: dict[str, int] = {}
        for tk in tasks:
            items = _read_jsonl(args.bench_dir / TASK_FILES[tk])
            out_lines = [
                json.dumps(
                    {
                        "task_id": it["task_id"],
                        "task": it["task"],
                        "image": it.get("image"),
                        "prompt": build_vlm_prompt(it),
                    },
                    sort_keys=True,
                )
                for it in items
            ]
            (args.out / f"prompts_{tk}.jsonl").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
            )
            exported[tk] = len(out_lines)
        report["exported_prompts"] = exported

    elif args.mode == "physics_ablation":
        ablations = list(ABLATION_PRESETS) if args.ablation == "all" else [args.ablation]
        phys_tasks = [tk for tk in tasks if tk in ("T2", "T3", "T4", "T5")]
        abl_reports: dict[str, dict] = {}
        for abl in ablations:
            per_task: dict[str, dict] = {}
            for tk in phys_tasks:
                items = _read_jsonl(args.bench_dir / TASK_FILES[tk])
                if not items:
                    continue
                preds = {
                    it["task_id"]: run_physics_posterior_baseline(
                        it, ablation=abl, n_conf=args.n_conf, seed=args.seed
                    )
                    for it in items
                }
                per_task[tk] = _score_task(tk, items, preds, args.seed)
            abl_reports[abl] = per_task
        report["ablations"] = abl_reports

    elif args.mode == "import_preds":
        if args.predictions is None:
            raise ValueError("--predictions is required for --mode import_preds")
        preds = import_external_ocsr_predictions(args.predictions)
        scored: dict[str, dict] = {}
        for tk in tasks:
            items = _read_jsonl(args.bench_dir / TASK_FILES[tk])
            if items:
                scored[tk] = _score_task(tk, items, preds, args.seed)
        report["tasks"] = scored

    elif args.mode == "live_vlm":
        scored = {}
        for tk in tasks:
            items = _read_jsonl(args.bench_dir / TASK_FILES[tk])
            if not items:
                continue
            preds = {}
            for it in items:
                img_p = (args.bench_dir / it["image"]) if it.get("image") else None
                img_arg = img_p if (img_p and img_p.exists()) else None
                resp = call_frontier_vlm(args.provider, args.model, build_vlm_prompt(it), img_arg)
                resp["task_id"] = it["task_id"]
                preds[it["task_id"]] = resp
            scored[tk] = _score_task(tk, items, preds, args.seed)
        report["tasks"] = scored

    (args.out / "eval_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
