"""Generate notebooks/00_colab_setup.ipynb (kept as a script so the notebook is reviewable)."""

import json
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


cells = [
    md(
        "# Ouroboros — Colab setup (A100)\n\n"
        "Run top to bottom on a GPU runtime. Mounts Drive, clones the repo, installs pinned deps "
        "(keeping Colab's CUDA PyTorch), copies WebDataset shards to local disk, and runs the test "
        "suite. All project code runs in subprocesses (`!python ...`) so that pinned numpy etc. "
        "take effect without restarting the kernel.\n\n"
        "If the repo is private, add a Colab secret `GITHUB_TOKEN` (key icon in the left sidebar) "
        "with read access to the repository."
    ),
    code(
        "import time, os, subprocess\n"
        "T0 = time.time()\n"
        "!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv\n"
        "!python --version"
    ),
    code(
        "# ---- configuration ----\n"
        "REPO = 'yaniguan/ouroboros-ocsr'\n"
        "BRANCH = 'claude/vigilant-johnson-j4882f'  # set to 'main' once merged\n"
        "DRIVE_ROOT = '/content/drive/MyDrive/ouroboros'  # checkpoints, shards, results live here\n"
        "LOCAL_SHARDS = '/content/shards'\n"
        "REPO_DIR = '/content/ouroboros-ocsr'"
    ),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "os.makedirs(DRIVE_ROOT, exist_ok=True)\n"
        "for sub in ('shards', 'runs', 'results'):\n"
        "    os.makedirs(f'{DRIVE_ROOT}/{sub}', exist_ok=True)"
    ),
    code(
        "token = None\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    token = userdata.get('GITHUB_TOKEN')\n"
        "except Exception:\n"
        "    pass\n"
        "url = f'https://{token}@github.com/{REPO}.git' if token else f'https://github.com/{REPO}.git'\n"
        "if os.path.isdir(REPO_DIR):\n"
        "    subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin', BRANCH], check=True)\n"
        "    subprocess.run(['git', '-C', REPO_DIR, 'checkout', '-B', BRANCH, f'origin/{BRANCH}'], check=True)\n"
        "else:\n"
        "    subprocess.run(['git', 'clone', '-b', BRANCH, url, REPO_DIR], check=True)\n"
        "!git -C {REPO_DIR} log --oneline -1"
    ),
    code(
        '%%bash -s "$REPO_DIR"\n'
        "set -e\n"
        'cd "$1"\n'
        "# py3nj (escnn dependency) builds from source and needs a Fortran compiler\n"
        "which gfortran || (apt-get -qq update && apt-get -qq install -y gfortran > /dev/null)\n"
        "pip install -q -r requirements-colab.txt\n"
        "pip install -q --no-deps -e .\n"
        "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
    ),
    code(
        "# Copy shards Drive -> local disk (Drive FUSE is too slow for training I/O).\n"
        "src = f'{DRIVE_ROOT}/shards'\n"
        "os.makedirs(LOCAL_SHARDS, exist_ok=True)\n"
        "if any(True for _ in os.scandir(src)):\n"
        "    !rsync -a --info=progress2 {src}/ {LOCAL_SHARDS}/\n"
        "else:\n"
        "    print('no shards on Drive yet; skipping copy')\n"
        "!du -sh {LOCAL_SHARDS}"
    ),
    code(
        '!cd {REPO_DIR} && python -c "import escnn, mace; from escnn import gspaces; '
        "from mace.calculators import mace_off; import rdkit; print('escnn/mace/rdkit import OK', rdkit.__version__)\""
    ),
    code("!cd {REPO_DIR} && python -m pytest -q"),
    code(
        "print(f'wall time: {(time.time() - T0) / 60:.1f} min')\n"
        "!pip freeze | grep -iE '^(torch|escnn|mace-torch|rdkit|numpy|e3nn)=='"
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

# notebook JSON wants each source as a list of lines
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

out = Path(__file__).resolve().parents[1] / "notebooks" / "00_colab_setup.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
