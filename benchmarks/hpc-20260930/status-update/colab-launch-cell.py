# Execute only this current-version cell after Drive is mounted.
from pathlib import Path
import hashlib, io, json, os, shutil, subprocess, sys, zipfile
from google.colab import files
uploaded = files.upload()
assert len(uploaded) == 1
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == "1d5bda895362a08adfad2068cff326bcef06097e5bc0af2d23b86e3c6a54e75f"
RUNROOT = Path("/content/ouroboros-20260930")
CODE = RUNROOT / "code"
CODE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for item in archive.infolist():
        target = (CODE / item.filename).resolve()
        assert target.is_relative_to(CODE.resolve())
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
        assert not target.exists(), str(target)
    archive.extractall(CODE)
DRIVE = Path("/content/drive/MyDrive/ouroboros")
assert (DRIVE / "data/full/shards").is_dir()
PERSIST = DRIVE / "colab-20260930-ac-s0"
PERSIST.mkdir(exist_ok=True)
(PERSIST / "source.zip").write_bytes(blob)
log = (PERSIST / "pipeline.log").open("a", buffering=1)
colab_job = subprocess.Popen([sys.executable, "-u", str(CODE / "scripts/hpc/colab_run.py"),
    "--root", str(RUNROOT), "--drive-root", str(DRIVE)],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("RUN_ID colab-20260930-ac-s0 PID", colab_job.pid)
print("Persistent output:", PERSIST)
