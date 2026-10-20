"""Run one v1 grid task (train, then evaluate) inside the Hoffman2 container.

    python scripts/hpc/v1_task.py --task 3 --original $OURO_ROOT --v0 $V0 --root $V1

Layout: ``$V1/code`` (this repository at the release commit), ``$V1/data/real`` (ingested real
shards: train/val/test), ``$V1/data/eval_keys.txt``; synthetic shards come from the audited v0 view
``$V0/data/full/shards``. Training is resumable: rerunning a task continues from ``last.pt``.

Checkpoint selection: synthetic-only tasks (fraction 0) use the synthetic validation set, exactly as
in v0; mixed tasks use the real dev split (article-disjoint from real train and test). The real test
split is never used for selection. Every task evaluates the rendered test set and the real test set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v1_grid import grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--v0", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    t = grid()[a.task - 1]
    root, code = a.root, a.root / "code"
    synth = a.v0 / "data/full/shards"
    real = root / "data/real"
    run = root / "runs" / t["run_id"]
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    mixed = t["fraction"] > 0
    overrides = [
        f"seed={t['seed']}",
        f"data.synthetic_root={synth}",
        f"data.exclude_keys={root / 'data/eval_keys.txt'}",
        f"data.real_fraction={t['fraction']}",
        f"data.real_roots=[{real}]" if mixed else "data.real_roots=[]",
        "data.num_workers=4",
        "train.device=cuda",
        f"validation.root={real if mixed else synth}",
        "validation.split=val",
        f"validation.samples={500 if mixed else 1000}",
        f"eval.sets.rendered_test.root={synth}",
        f"eval.sets.real_test.root={real}",
        "eval.sets.real_test.split=test",
        "eval.sets.real_test.kind=real",
        "eval.batch_size=32",
    ]
    py = str(a.original / "venv/bin/python")
    env = dict(
        os.environ,
        PYTHONPATH=str(code),
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    status_path = reports / f"status-{t['run_id']}.json"
    state = {"task": t, "job_id": os.environ.get("JOB_ID"), "stages": [], "started": time.time()}

    def save(stage: str, outcome: str) -> None:
        state.update(stage=stage, status=outcome, updated=time.time())
        status_path.write_text(json.dumps(state, indent=1))

    def run_stage(stage: str, command: list[str]) -> None:
        save(stage, "running")
        started = time.time()
        with (reports / f"{t['run_id']}-{stage}.log").open("a") as log:
            result = subprocess.run(
                command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        state["stages"].append(
            {"stage": stage, "exit_code": result.returncode, "seconds": time.time() - started}
        )
        if result.returncode:
            save(stage, "failed")
            raise SystemExit(f"{stage} exit {result.returncode}")
        save(stage, "passed")

    run_stage(
        "native-bf16",
        [
            py,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "assert torch.cuda.is_bf16_supported(including_emulation=False); "
            "print(torch.cuda.get_device_name())",
        ],
    )
    cfg = f"configs/{t['config']}"
    train = [py, "-m", "ouroboros.train", "--config", cfg, "--out", str(run), "--set", *overrides]
    run_stage("train", train)
    evaluate = [py, "scripts/evaluate.py", "--run", str(run), "--set", *overrides]
    run_stage("evaluate", evaluate)
    save("done", "complete")


if __name__ == "__main__":
    main()
