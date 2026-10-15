"""Derive a clean pilot without changing original shards, then audit/resume/train A/C."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    original = Path("/content/ouroboros-20260930")
    root = Path("/content/ouroboros-20260930-clean3")
    persistent = Path("/content/drive/MyDrive/ouroboros/colab-20260930-ac-s0/clean3")
    expected = {"train_00011370", "train_00057670", "train_00167450"}
    diag_file = original / "reports/data-audit-diagnostic.json"
    diag = json.loads(diag_file.read_text())
    assert {x["key"] for x in diag["stereo_mismatches"]} == expected
    assert all(x["split"] == "train" for x in diag["stereo_mismatches"])
    source = json.loads((original / "code/colab-source.json").read_text())
    for entry in source["files"]:
        assert sha(original / "code" / entry["path"]) == entry["sha256"]
    root.mkdir(exist_ok=False)
    persistent.mkdir(exist_ok=False)
    for name in ("reports", "logs", "runs"):
        (persistent / name).mkdir()
        (root / name).symlink_to(persistent / name, target_is_directory=True)
    for name in ("code", "venv", "cache"):
        (root / name).symlink_to(original / name, target_is_directory=True)
    reports = root / "reports"
    state = {
        "run_id": "colab-20260930-ac-s0-clean3",
        "pid": os.getpid(),
        "stages": [],
        "started": time.time(),
        "excluded": sorted(expected),
    }

    def status(stage, outcome, **extra):
        state.update(stage=stage, status=outcome, updated=time.time(), **extra)
        tmp = reports / "pipeline-status.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(reports / "pipeline-status.json")
        print(stage, outcome, extra, flush=True)

    try:
        status("derive-clean-view", "running")
        receipts = json.loads((original / "reports/data-transfer-receipts.json").read_text())
        new_receipts = {
            "scope": "original pilot with exactly three stereo-unstable train pairs quarantined",
            "files": [],
        }
        dropped_pairs, affected = set(), []
        for item in receipts["files"]:
            rel = Path(item["path"])
            src, dst = original / "data/full" / rel, root / "data/full" / rel
            assert src.stat().st_size == item["bytes"] and sha(src) == item["sha256"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel == Path("manifests/train.tsv"):
                with src.open() as reader, dst.open("w", newline="") as writer:
                    rows = csv.DictReader(reader, delimiter="\t")
                    out = csv.DictWriter(writer, fieldnames=rows.fieldnames, delimiter="\t")
                    out.writeheader()
                    for row in rows:
                        if f"train_{int(row['idx']):08d}" not in expected:
                            out.writerow(row)
            elif rel.name in {"train-000011.tar", "train-000057.tar", "train-000167.tar"}:
                with tarfile.open(src) as inp, tarfile.open(dst, "w") as out:
                    removed = []
                    for member in inp:
                        key = member.name.rsplit(".", 1)[0]
                        if key in expected:
                            removed.append(member.name)
                            dropped_pairs.add(key)
                        else:
                            out.addfile(member, inp.extractfile(member))
                    assert len(removed) == 2
                affected.append(
                    {
                        "path": str(rel),
                        "removed_members": removed,
                        "original_sha256": item["sha256"],
                        "derived_sha256": sha(dst),
                    }
                )
            else:
                os.link(src, dst)
            new_receipts["files"].append(
                {"path": str(rel), "bytes": dst.stat().st_size, "sha256": sha(dst)}
            )
        assert dropped_pairs == expected
        quarantine = {
            "diagnostic_sha256": sha(diag_file),
            "source": source,
            "excluded_records": diag["stereo_mismatches"],
            "affected_shards": affected,
            "original_preserved": True,
            "validation_and_test_unchanged": True,
        }
        (reports / "quarantine.json").write_text(json.dumps(quarantine, indent=2) + "\n")
        (reports / "data-transfer-receipts.json").write_text(
            json.dumps(new_receipts, indent=2) + "\n"
        )
        (reports / "data-transfer.ok").touch()
        # Same interpreter, hardware and source: retain the passed preflight evidence.
        for name in (
            "setup.ok",
            "tests.ok",
            "capacity-A.ok",
            "capacity-C.ok",
            "environment.txt",
            "capacity-A.json",
            "capacity-C.json",
        ):
            shutil.copyfile(original / "reports" / name, reports / name)
        shutil.copyfile(original / "logs/tests.log", root / "logs/tests.log")
        status("derive-clean-view", "passed")
        env = dict(os.environ, OURO_ROOT=str(root), MPLBACKEND="Agg", PYTHONUNBUFFERED="1")

        def work(stage, task, index=1):
            status(stage, "running")
            started = time.time()
            with (root / "logs" / f"{stage}.log").open("a") as log:
                result = subprocess.run(
                    ["bash", "scripts/hpc/work.sh"],
                    cwd=root / "code",
                    env=env | {"OURO_TASK": task, "SGE_TASK_ID": str(index)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            state["stages"].append(
                {"stage": stage, "exit_code": result.returncode, "seconds": time.time() - started}
            )
            if result.returncode:
                raise RuntimeError(f"{stage}: exit {result.returncode}")
            status(stage, "passed")

        work("data-audit", "audit")
        audit = json.loads((reports / "data-audit.json").read_text())
        assert audit["passed"] and audit["selections"]["train"]["count"] == 200000
        for split in ("val", "test"):
            assert audit["selections"][split] == diag["selections"][split]
        for arm, index in (("A", 1), ("C", 2)):
            work(f"resume-{arm}", "resume", index)
            shutil.copyfile(root / f"resume-{arm}/report.json", reports / f"resume-{arm}.json")
        for arm, index in (("A", 1), ("C", 2)):
            work(f"train-{arm}-s0", "pilot", index)
        status("training", "complete")
    except BaseException as exc:
        status(state.get("stage", "startup"), "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
