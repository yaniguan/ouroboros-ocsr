"""50-step dry run of every sweep config (Phase 4 / Am1-E DoD), locally at reduced resolution.

Architecture, optimizer, mixture sampler and augmentation are exactly as configured; only the
resolution, batch size, worker count, AMP and data roots are overridden for the CPU VM. Configs with
real_fraction > 0 use the STAND-IN real shards (scripts/make_standin_real.py) — pipeline check only.
One run per arm is additionally evaluated (angle 0 + a 4-angle sweep on a handful of samples) and
all evaluated runs are aggregated, so train -> evaluate -> aggregate is exercised end to end.

    python scripts/dry_run_sweep.py --configs configs/sweep --out /tmp/dry_sweep
Writes benchmarks/sweep/dry_run.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ouroboros.train.config import load_config  # noqa: E402
from ouroboros.train.trainer import Trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs/sweep")
    ap.add_argument("--out", default="/tmp/dry_sweep")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--synthetic-root", default="data/full/shards")
    ap.add_argument("--real-root", default="data/standin_real/shards")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pattern", default="*.yaml", help="glob of configs to run")
    ap.add_argument(
        "--prior",
        default=None,
        help="JSONL of per-config records from an earlier "
        "(interrupted) dry run to merge; those configs are skipped",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, evaluated = [], set()
    if a.prior:
        results = [json.loads(x) for x in Path(a.prior).read_text().splitlines() if x.strip()]
        results = [r for r in results if r.get("ok")]
        evaluated = {r["arm"] for r in results if r.get("evaluated")}
    done = {r["config"] for r in results}
    cfgs = [p for p in sorted(Path(a.configs).glob(a.pattern)) if p.name not in done][: a.limit]
    for path in cfgs:
        raw = yaml.safe_load(path.read_text())
        frac = raw["data"]["real_fraction"]
        overrides = [
            f"data.synthetic_root={a.synthetic_root}",
            f"data.image_size={a.image_size}",
            "data.num_workers=0",
            "train.batch_size=4",
            f"train.steps={a.steps}",
            "train.warmup=10",
            "train.amp=none",
            "train.log_every=10",
            f"train.ckpt_every={a.steps}",
            f"eval.sets.rendered_test.root={a.synthetic_root}",
            "eval.sets.rendered_test.max_samples=6",
            "eval.sweep_max_samples=3",
            "eval.sweep_angles_step=90",
            "eval.batch_size=6",
        ]
        if frac > 0:
            overrides.append(f"data.real_roots=[{a.real_root}]")
        cfg = load_config(path, overrides)
        cfg["eval"]["sets"]["standin_real_test"] = {
            "root": a.real_root,
            "split": "test",
            "kind": "real",
            "max_samples": 6,
        }
        run_dir = out / path.stem
        t0 = time.time()
        rec = {"config": path.name, "arm": raw["meta"]["arm"], "real_fraction": frac}
        try:
            tr = Trainer(cfg, run_dir)
            tr.fit()
            logs = [json.loads(x) for x in (run_dir / "log.jsonl").read_text().splitlines()]
            losses = [r["loss"] for r in logs if "loss" in r]
            rec.update(
                ok=tr.step == a.steps,
                steps=tr.step,
                first_loss=losses[0],
                last_loss=losses[-1],
                n_real=len(tr.real) if tr.real else 0,
            )
            if raw["meta"]["arm"] not in evaluated:
                import evaluate

                evaluate.main(["--run", str(run_dir)])
                evaluated.add(raw["meta"]["arm"])
                rec["evaluated"] = True
        except Exception as e:  # noqa: BLE001 - record and continue
            rec.update(ok=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
        rec["sec"] = round(time.time() - t0, 1)
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "tb"}), flush=True)
        if not rec.get("evaluated"):
            shutil.rmtree(run_dir, ignore_errors=True)  # checkpoints are large
        else:
            shutil.rmtree(run_dir / "ckpt", ignore_errors=True)
    import aggregate

    eval_runs = [str(out / r["config"][:-5]) for r in results if r.get("evaluated")]
    if eval_runs:
        aggregate.main(["--runs", *eval_runs, "--out", str(out / "aggregate")])
    summary = {
        "n_configs": len(results),
        "n_ok": sum(r.get("ok", False) for r in results),
        "failures": [r for r in results if not r.get("ok")],
        "evaluated_arms": sorted(evaluated),
        "overrides": f"image_size={a.image_size}, batch 4, {a.steps} steps, fp32, CPU",
        "results": [{k: v for k, v in r.items() if k != "tb"} for r in results],
        "date": time.strftime("%Y-%m-%d"),
    }
    dest = ROOT / "benchmarks" / "sweep"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "dry_run.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['n_ok']}/{summary['n_configs']} configs completed {a.steps} steps")


if __name__ == "__main__":
    main()
