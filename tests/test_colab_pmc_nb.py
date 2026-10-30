"""colab/finetune_real_pmc.ipynb: cells compile, failing subprocess output is shown and saved."""

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

NB = Path("colab/finetune_real_pmc.ipynb")


def _codes():
    nb = json.loads(NB.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_compile():
    for src in _codes():
        ast.parse(src)


def test_run_helper_saves_tail_on_failure(tmp_path, capsys):
    first = _codes()[0]
    src = first[first.index("def shell") : first.index("def sha256")]
    g = {"Path": Path, "subprocess": subprocess, "time": time, "DRIVE_ROOT": tmp_path,
         "PY": sys.executable, "REPO": Path.cwd()}
    exec(src, g)
    code = "import sys; print('to-stdout'); print('to-stderr', file=sys.stderr); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError):
        g["run"]("-c", code, log=tmp_path / "train_stderr.log")
    out = capsys.readouterr().out
    assert "FAILED (exit 3)" in out and "to-stderr" in out.split("FAILED")[1]
    saved = (tmp_path / "train_stderr.log").read_text()
    assert "to-stdout" in saved and "to-stderr" in saved and "[exit status 3]" in saved
    assert g["run"]("-c", code, check=False).returncode == 3  # default log kept on failure
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1
    g["run"]("-c", "print('ok')")  # default log removed on success
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1


def test_train_cell_skips_finished_runs_and_logs():
    train = next(c for c in _codes() if "ouroboros.train" in c and "DONE_CHECK" in c)
    assert "if train_done(r['dir'])" in train
    assert "log=r['dir'] / 'train_stderr.log'" in train
