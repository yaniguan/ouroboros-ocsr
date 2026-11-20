"""Distill the OCSR teacher into a small edge student: check the config, print the plan.

    python -m lab.edge_plan --dry-run                      # default config
    python -m lab.edge_plan lab/configs/edge_distill.json --dry-run --json

Only ``--dry-run`` exists today. It runs nothing, loads no model and downloads nothing.
Each step is marked ``exists`` (an existing repo command) or ``todo`` (code still to write).

Teacher: the fine-tuned A + C ensemble with beam search and TTA (the strongest decoding in
benchmarks/finetune-v2-oct30/). Student: one arm, greedy, then quantized for an edge box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "edge_distill.json"
ARMS = ("A", "B", "C")
DECODING = ("greedy", "beam")
QUANT = ("none", "fp16", "int8-dynamic")
DEVICES = ("cpu", "jetson-orin-nano-super", "jetson-agx-orin-64gb", "gpu-24gb")
TTA = ("pad", "sharpen")  # same views as scripts/predict_real.py
# Repo rule (DATA.md): test sets are never used to select checkpoints or limits.
TEST_SET_MARKERS = ("subset100", "frozen", "dev-test", "test")


class PlanError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate(cfg: dict) -> dict:
    """Raise PlanError with every problem found; return cfg unchanged if fine."""
    errs = []
    t, s, pl = cfg.get("teacher", {}), cfg.get("student", {}), cfg.get("pseudo_labels", {})
    sel, q, tgt = cfg.get("selection", {}), cfg.get("quantize", {}), cfg.get("target", {})
    if not t.get("checkpoints"):
        errs.append("teacher.checkpoints: need at least one checkpoint")
    if not isinstance(t.get("beam", 1), int) or t.get("beam", 1) < 1:
        errs.append("teacher.beam: must be an integer >= 1")
    bad_tta = [v for v in t.get("tta", []) if v not in TTA]
    if bad_tta:
        errs.append(f"teacher.tta: unknown views {bad_tta}; allowed {list(TTA)}")
    if s.get("arm") not in ARMS:
        errs.append(f"student.arm: must be one of {list(ARMS)}")
    if s.get("decoding", "greedy") not in DECODING:
        errs.append(f"student.decoding: must be one of {list(DECODING)}")
    m = pl.get("min_margin")
    if m is not None and (not isinstance(m, int | float) or not math.isfinite(m) or m < 0):
        errs.append("pseudo_labels.min_margin: must be finite and >= 0")
    d = pl.get("max_disagreement")
    if d is not None and (not isinstance(d, int | float) or not 0 <= d <= 1):
        errs.append("pseudo_labels.max_disagreement: must be in [0, 1]")
    for key, block in (("pseudo_labels", pl), ("selection", sel)):
        where = f"{block.get('shards', '')}:{block.get('split', '')}".lower()
        if any(mark in where for mark in TEST_SET_MARKERS):
            errs.append(f"{key}: uses a test set ({where}); use train/val data only")
    if not cfg.get("leakage_eval_sets"):
        errs.append("leakage_eval_sets: list the eval sets the pseudo-labels must not overlap")
    if q.get("method", "none") not in QUANT:
        errs.append(f"quantize.method: must be one of {list(QUANT)}")
    if tgt.get("device") not in DEVICES:
        errs.append(f"target.device: must be one of {list(DEVICES)}")
    if errs:
        raise PlanError("; ".join(errs))
    return cfg


def plan(cfg: dict) -> list[dict]:
    """Ordered steps. Commands use real repo scripts where they exist."""
    t, s, pl = cfg["teacher"], cfg["student"], cfg["pseudo_labels"]
    sel, q, tgt = cfg["selection"], cfg.get("quantize", {}), cfg["target"]
    run = f"runs/{cfg.get('name', 'edge')}"
    ckpts = " ".join(f"--ckpt {c}" for c in t["checkpoints"])
    tta = f" --tta {','.join(t['tta'])}" if t.get("tta") else ""
    gate = ""
    if pl.get("min_margin") is not None:
        gate += f" --min-margin {pl['min_margin']}"
    if pl.get("max_disagreement") is not None:
        gate += f" --max-disagreement {pl['max_disagreement']}"
    student_ckpt = f"{run}/student/ckpt/best-step-<N>.pt"
    pseudo = f"{run}/pseudo"
    roots = ",".join([*s.get("real_roots", []), f"{pseudo}/shards"])
    beam = "" if s.get("decoding", "greedy") == "greedy" else " --beam 5"
    steps = [
        dict(name="choose gate limits on validation", status="exists",
             cmd=f"python scripts/predict_real.py {ckpts} --beam {t.get('beam', 1)}{tta} "
                 f"--shards {sel['shards']} --split {sel['split']} --out {run}/val_teacher.csv "
                 f"--nbest-out {run}/val_teacher.jsonl && python scripts/eval_selective.py "
                 f"{run}/val_teacher.jsonl --margins 0 0.2 0.5"),
        dict(name="teacher pseudo-labels on unlabeled figures", status="exists",
             cmd=f"python scripts/predict_real.py {ckpts} --beam {t.get('beam', 1)}{tta}{gate} "
                 f"--unlabeled --shards {pl['shards']} --split {pl['split']} "
                 f"--out {run}/teacher.csv"),
        dict(name="keep usable rows -> pseudo-label manifest -> shards", status="exists",
             cmd=f"python -m lab.molecule_context {run}/teacher.csv --out {run}/teacher.jsonl "
                 f"--manifest-out {pseudo}/manifest.csv "
                 f"&& python scripts/ingest_real.py --manifest {pseudo}/manifest.csv "
                 f"--out {pseudo}/shards"),
        dict(name="leakage gate vs all eval sets", status="exists",
             cmd="python scripts/check_leakage.py --eval "
                 + " ".join(cfg["leakage_eval_sets"])
                 + f" --train pseudo={pseudo}/manifest.csv --fail-on-connectivity"),
        dict(name=f"train student arm {s['arm']}", status="exists",
             cmd=f"python -m ouroboros.train --config {s['config']} --out {run}/student "
                 f"--set train.init_checkpoint={s.get('init', '')} "
                 f"data.real_roots=[{roots}]"),
        dict(name="score student on validation", status="exists",
             cmd=f"python scripts/predict_real.py --ckpt {student_ckpt}{beam} "
                 f"--shards {sel['shards']} --split {sel['split']} --out {run}/val_student.csv "
                 f"&& python scripts/eval_predictions.py {run}/val_student.csv "
                 f"--json {run}/val_student.json"),
        dict(name=f"quantize ({q.get('method', 'none')})", status="exists",
             cmd=f"python -m lab.quantize {student_ckpt} --method {q.get('method', 'none')} "
                 f"--shards {sel['shards']} --split {sel['split']} "
                 f"--report {run}/val_quantized.json"),
        dict(name=f"latency on {tgt['device']}", status="exists",
             cmd=f"python scripts/profile_infer.py --ckpt {student_ckpt} --shards {sel['shards']} "
                 f"--split {sel['split']} --beam 1 --name student --out-dir {run}/profile"),
    ]
    for i, step in enumerate(steps, 1):
        step["step"] = i
    return steps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    p.add_argument("--dry-run", action="store_true", help="check the config and print the plan")
    p.add_argument("--json", action="store_true", help="print the plan as JSON")
    a = p.parse_args(argv)
    if not a.dry_run:
        print("Only --dry-run is implemented. Nothing was run.", file=sys.stderr)
        return 2
    try:
        cfg = validate(load_config(a.config))
    except PlanError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    steps = plan(cfg)
    if a.json:
        print(json.dumps({"name": cfg.get("name"), "dry_run": True, "steps": steps}, indent=1))
        return 0
    print(f"plan {cfg.get('name')} (dry run, nothing executed; no metrics yet)")
    for st in steps:
        print(f"{st['step']}. [{st['status']}] {st['name']}\n   $ {st['cmd']}")
        if st.get("todo"):
            print(f"   todo: {st['todo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
