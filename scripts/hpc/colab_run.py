"""Run the current synthetic-only A/C pilot on an already allocated Colab GPU.

Use the notebook to mount Drive and upload the reviewed source bundle first.
All gates fail closed. Training checkpoints and logs are written to Drive.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    args = parser.parse_args()
    root, drive = args.root.resolve(), args.drive_root.resolve()
    code = root / "code"
    if not (drive / "data/full/shards").is_dir():
        raise RuntimeError("Mount the existing project Drive before launching")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    persistent = drive / "colab-20260930-ac-s0"
    persistent.mkdir(exist_ok=True)
    for name in ("reports", "runs", "logs"):
        target = persistent / name
        target.mkdir(exist_ok=True)
        local = root / name
        if not local.exists():
            local.symlink_to(target, target_is_directory=True)
        if local.resolve() != target:
            raise RuntimeError(f"Unexpected output location: {local}")
    reports = root / "reports"
    manifest = json.loads((code / "colab-source.json").read_text())
    for item in manifest["files"]:
        if digest(code / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Source mismatch: {item['path']}")
    pinned = persistent / "source.json"
    if pinned.exists() and json.loads(pinned.read_text()) != manifest:
        raise RuntimeError("Persistent run belongs to a different source snapshot")
    pinned.write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, OURO_ROOT=str(root), MPLBACKEND="Agg",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", PYTHONPATH=str(code))
    state = {"run_id": "colab-20260930-ac-s0", "pid": os.getpid(),
             "started": time.time(), "persistent": str(persistent), "stages": []}

    def status(stage, outcome, **extra):
        state.update(stage=stage, status=outcome, updated=time.time(), **extra)
        temporary = reports / "pipeline-status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(reports / "pipeline-status.json")
        print(stage, outcome, extra, flush=True)

    def run(stage, command, overrides=None):
        status(stage, "running")
        started = time.time()
        with (root / "logs" / f"{stage}.log").open("a", buffering=1) as log:
            log.write(f"\nSTART {time.ctime()} {command!r}\n")
            result = subprocess.run(command, cwd=code, env=env | (overrides or {}),
                                    stdout=log, stderr=subprocess.STDOUT)
        state["stages"].append({"stage": stage, "exit_code": result.returncode,
                                "seconds": time.time() - started})
        if result.returncode:
            raise RuntimeError(f"{stage} exited {result.returncode}; see logs/{stage}.log")
        status(stage, "passed")

    def work(stage, task, index=1):
        run(stage, ["bash", "scripts/hpc/work.sh"],
            {"OURO_TASK": task, "SGE_TASK_ID": str(index)})

    try:
        run("hardware", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                         "--format=csv"])
        if not (reports / "setup.ok").exists():
            run("setup", ["bash", "scripts/hpc/setup.sh"])
        py = str(root / "venv/bin/python")
        run("native-bf16", [py, "-c", "import torch,escnn,mace; "
                            "assert torch.cuda.is_available(); "
                            "assert torch.cuda.is_bf16_supported(including_emulation=False); "
                            "print(torch.__version__,torch.cuda.get_device_name())"])
        status("data-copy", "running")
        receipt_path = code / "colab-data-receipts.json"
        receipt = json.loads(receipt_path.read_text())
        for i, item in enumerate(receipt["files"], 1):
            src = drive / "data/full" / item["path"]
            dst = root / "data/full" / item["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or digest(dst) != item["sha256"]:
                temp = dst.with_suffix(dst.suffix + ".copying")
                shutil.copyfile(src, temp)
                if temp.stat().st_size != item["bytes"] or digest(temp) != item["sha256"]:
                    raise RuntimeError(f"Data mismatch: {item['path']}")
                temp.replace(dst)
            if i % 25 == 0:
                status("data-copy", "running", copied_files=i)
        shutil.copyfile(receipt_path, reports / "data-transfer-receipts.json")
        (reports / "data-transfer.ok").touch()
        status("data-copy", "passed", copied_files=len(receipt["files"]))
        if not (reports / "tests.ok").exists():
            work("tests", "tests")
        if not (reports / "data-audit.ok").exists():
            work("data-audit", "audit")
        # A/C are trained only after both arms pass the same prerequisite gates.
        for arm, capacity_index, resume_index in (("A", 1, 1), ("C", 3, 2)):
            if not (reports / f"capacity-{arm}.ok").exists():
                work(f"capacity-{arm}", "capacity", capacity_index)
            if not (reports / f"resume-{arm}.ok").exists():
                work(f"resume-{arm}", "resume", resume_index)
                shutil.copyfile(root / f"resume-{arm}/report.json",
                                reports / f"resume-{arm}.json")
        for arm, index in (("A", 1), ("C", 2)):
            if not (reports / f"train-{arm}-s0.ok").exists():
                work(f"train-{arm}-s0", "pilot", index)
        status("training", "complete")
    except BaseException as exc:
        status(state.get("stage", "startup"), "failed", error=str(exc))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
