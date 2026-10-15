"""Restore the audited pilot byte-for-byte, then gate B training and evaluation.

Run inside the verified Hoffman2 container, with the archived A/C source bundle.
This adds rotation augmentation only; it never edits A/C runs or original data.
"""

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    root, original = a.root, a.original
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    state = {"job_id": os.environ.get("JOB_ID"), "started": time.time(), "stages": []}

    def status(stage, outcome, **extra):
        state.update(stage=stage, status=outcome, updated=time.time(), **extra)
        tmp = reports / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(reports / "status.json")
        print(stage, outcome, extra, flush=True)

    try:
        status("source", "running")
        archive = root / "source.zip"
        assert sha(archive) == "1d5bda895362a08adfad2068cff326bcef06097e5bc0af2d23b86e3c6a54e75f"
        code = root / "code"
        code.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                assert (code / item.filename).resolve().is_relative_to(code.resolve())
                assert (item.external_attr >> 16) & 0o170000 != 0o120000
            z.extractall(code)
        for entry in json.loads((code / "colab-source.json").read_text())["files"]:
            assert sha(code / entry["path"]) == entry["sha256"]
        (code / "configs/pilot-B-200k.yaml").write_text(
            "base: pilot-A-200k.yaml\ntrain:\n  rotation_aug: true\n"
        )
        py = str(original / "venv/bin/python")
        env = dict(
            os.environ,
            PYTHONPATH=str(code),
            MPLBACKEND="Agg",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )

        def run(stage, command):
            status(stage, "running")
            started = time.time()
            with (reports / f"{stage}.log").open("a") as log:
                result = subprocess.run(
                    command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            state["stages"].append(
                {"stage": stage, "exit_code": result.returncode, "seconds": time.time() - started}
            )
            if result.returncode:
                raise RuntimeError(f"{stage} exit {result.returncode}")
            status(stage, "passed")

        run(
            "native-bf16",
            [
                py,
                "-c",
                "import torch; assert torch.cuda.is_available(); "
                "assert torch.cuda.is_bf16_supported(including_emulation=False); "
                "print(torch.__version__, torch.cuda.get_device_name())",
            ],
        )
        run("tests", [py, "-m", "pytest", "-q"])
        status("derive-clean-view", "running")
        receipts = json.loads((code / "colab-data-receipts.json").read_text())
        expected = json.loads((root / "expected-receipts.json").read_text())
        expected_by_path = {entry["path"]: entry for entry in expected["files"]}
        exclude = {"train_00011370", "train_00057670", "train_00167450"}
        removed = set()
        for item in receipts["files"]:
            rel = Path(item["path"])
            src, dst = original / "data/full" / rel, root / "data/full" / rel
            assert src.stat().st_size == item["bytes"] and sha(src) == item["sha256"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel == Path("manifests/train.tsv"):
                with src.open() as inp, dst.open("w", newline="") as out:
                    rows = csv.DictReader(inp, delimiter="\t")
                    writer = csv.DictWriter(out, fieldnames=rows.fieldnames, delimiter="\t")
                    writer.writeheader()
                    for row in rows:
                        if f"train_{int(row['idx']):08d}" not in exclude:
                            writer.writerow(row)
            elif rel.name in {"train-000011.tar", "train-000057.tar", "train-000167.tar"}:
                with tarfile.open(src) as inp, tarfile.open(dst, "w") as out:
                    for member in inp:
                        key = member.name.rsplit(".", 1)[0]
                        if key in exclude:
                            removed.add(key)
                        else:
                            out.addfile(member, inp.extractfile(member))
            elif not dst.exists():
                try:
                    os.link(src, dst)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    # Separate container bind mounts may prohibit hard links even
                    # when both paths live on the same underlying scratch volume.
                    temporary = dst.with_suffix(dst.suffix + ".copying")
                    shutil.copyfile(src, temporary)
                    temporary.replace(dst)
            want = expected_by_path[item["path"]]
            assert dst.stat().st_size == want["bytes"] and sha(dst) == want["sha256"]
        assert removed == exclude
        shutil.copyfile(root / "expected-receipts.json", reports / "data-transfer-receipts.json")
        (reports / "data-transfer.ok").touch()
        run("audit", [py, "scripts/hpc/audit_pilot.py", "--root", str(root)])
        audited = json.loads((reports / "data-audit.json").read_text())
        prior = json.loads((root / "expected-audit.json").read_text())
        assert audited["selections"] == prior["selections"]
        run(
            "capacity-B",
            [
                py,
                "scripts/benchmark_cuda.py",
                "--config",
                "configs/pilot-B-200k.yaml",
                "--steps",
                "50",
                "--warmup",
                "5",
                "--batch",
                "32",
                "--sequence-length",
                "128",
                "--output",
                str(reports / "capacity-B.json"),
            ],
        )
        data = str(root / "data/full/shards")
        overrides = [
            f"data.synthetic_root={data}",
            f"validation.root={data}",
            f"eval.sets.rendered_test.root={data}",
            "data.num_workers=4",
            "train.device=cuda",
            "eval.batch_size=32",
        ]
        run(
            "resume-B",
            [
                py,
                "scripts/benchmark_cuda_resume.py",
                "--config",
                "configs/pilot-B-200k.yaml",
                "--output",
                str(root / "resume-B"),
                "--checkpoint-step",
                "25",
                "--compared-steps",
                "100",
                "--set",
                *overrides,
                "validation.every=25",
                "validation.samples=32",
                "validation.patience=20",
            ],
        )
        run_dir = root / "runs/B_n200k_f0_s0"
        run(
            "train-B",
            [
                py,
                "-m",
                "ouroboros.train",
                "--config",
                "configs/pilot-B-200k.yaml",
                "--out",
                str(run_dir),
                "--set",
                *overrides,
            ],
        )
        run("evaluate-B", [py, "scripts/evaluate.py", "--run", str(run_dir)])
        status("B-pilot", "complete")
    except BaseException as exc:
        status(state.get("stage", "startup"), "failed", error=str(exc))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
