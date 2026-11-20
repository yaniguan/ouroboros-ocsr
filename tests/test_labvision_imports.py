import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_decode_import_pulls_no_torch_or_escnn():
    code = (
        "import sys, ouroboros.labvision.decode, ouroboros.labvision.render, "
        "ouroboros.labvision.wellstate\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'escnn', 'cv2', 'scipy')]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
