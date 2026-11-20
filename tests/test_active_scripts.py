"""The P2 scripts must start from a fresh checkout without PYTHONPATH."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["run_active_distill.py", "scan_mlip_blindspots.py"])
def test_script_help_runs(script):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "usage" in res.stdout.lower()
