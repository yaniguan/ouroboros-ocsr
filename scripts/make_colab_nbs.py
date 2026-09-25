"""Generate the Colab notebooks in notebooks/ (kept as a script so notebooks diff cleanly).

    python scripts/make_colab_nbs.py

00_colab_setup.ipynb     Phase 0 smoke run: Drive, clone, install, copy shards, pytest.
01_generate_data.ipynb   Phase 1: build the 1M-train dataset (200k = its prefix) and store on Drive.
02_train_eval.ipynb      Train one sweep config (resumable) and evaluate it; results on Drive.
03_geometry.ipynb        MACE-OFF benchmarks on 1,000 molecules + energy-error propagation.
"""

import json
from pathlib import Path

BRANCH = "claude/vigilant-johnson-j4882f"


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


def setup_cells(title: str, intro: str) -> list:
    return [
        md(
            f"# Ouroboros — {title}\n\n{intro}\n\n"
            "All project code runs in subprocesses (`!python ...`) so the pinned numpy etc. take "
            "effect without restarting the kernel. If the repo is private, add a Colab secret "
            "`GITHUB_TOKEN` (key icon in the left sidebar) with read access to the repository."
        ),
        code(
            "import time, os, subprocess, json\n"
            "T0 = time.time()\n"
            "!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true\n"
            "!python --version && nproc && free -g | head -2"
        ),
        code(
            "# ---- configuration ----\n"
            "REPO = 'yaniguan/ouroboros-ocsr'\n"
            f"BRANCH = '{BRANCH}'  # set to 'main' once merged\n"
            "DRIVE_ROOT = '/content/drive/MyDrive/ouroboros'  # data, runs, results live here\n"
            "REPO_DIR = '/content/ouroboros-ocsr'\n"
            "LOCAL_DATA = '/content/data/full'  # configs expect shards at /content/data/full/shards"
        ),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "for sub in ('data', 'runs', 'results', 'real'):\n"
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
            "os.chdir(REPO_DIR)\n"
            "!git log --oneline -1"
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
    ]


def copy_shards_cell(n_train_shards: str) -> dict:
    return code(
        "# Copy shards Drive -> local disk (Drive FUSE is too slow for training I/O).\n"
        f"N_TRAIN_SHARDS = {n_train_shards}  # 1000 samples per shard; None = all\n"
        "src = f'{DRIVE_ROOT}/data/full/shards'\n"
        "dst = f'{LOCAL_DATA}/shards'\n"
        "os.makedirs(dst, exist_ok=True)\n"
        "names = sorted(os.listdir(src)) if os.path.isdir(src) else []\n"
        "keep = [n for n in names if n.endswith('.tar') and (not n.startswith('train-') or\n"
        "        N_TRAIN_SHARDS is None or int(n[6:12]) < N_TRAIN_SHARDS)]\n"
        "t = time.time()\n"
        "for n in keep:\n"
        "    if not os.path.exists(f'{dst}/{n}'):\n"
        "        subprocess.run(['cp', f'{src}/{n}', f'{dst}/{n}'], check=True)\n"
        "print(f'{len(keep)} shards in {time.time() - t:.0f}s' if keep else 'no shards on Drive yet')\n"
        "!du -sh {dst}"
    )


def nb00() -> list:
    cells = setup_cells(
        "Colab setup (A100)",
        "Phase 0 smoke run: mounts Drive, clones the repo, installs pinned deps (keeping Colab's "
        "CUDA PyTorch), copies WebDataset shards to local disk and runs the test suite.",
    )
    cells += [
        copy_shards_cell("None"),
        code(
            '!python -c "import escnn, mace; from escnn import gspaces; '
            "from mace.calculators import mace_off; import rdkit; print('escnn/mace/rdkit import OK', rdkit.__version__)\""
        ),
        code("!python -m pytest -q"),
        code(
            "print(f'wall time: {(time.time() - T0) / 60:.1f} min')\n"
            "!pip freeze | grep -iE '^(torch|escnn|mace-torch|rdkit|numpy|e3nn)=='"
        ),
    ]
    return cells


def nb01() -> list:
    cells = setup_cells(
        "generate the 1M dataset (Phase 1)",
        "Builds pool -> composition -> shards for 1M training molecules (the 200k set is its "
        "first 200 shards) plus the shared 5k val / 10k test sets, then copies everything to "
        "Drive. CPU-only work: any runtime with many vCPUs works; expected ~35-60 min on 12 vCPUs, "
        "~7 GB of shards.",
    )
    cells += [
        code(
            "t = time.time()\n"
            "!python scripts/build_dataset.py --out {LOCAL_DATA} --cache /content/data_cache --preset 1m --workers $(nproc)\n"
            "GEN_MIN = (time.time() - t) / 60\n"
            "print(f'generation wall time: {GEN_MIN:.1f} min')"
        ),
        code(
            "t = time.time()\n"
            "!mkdir -p {DRIVE_ROOT}/data/full\n"
            "!rsync -a --info=progress2 {LOCAL_DATA}/ {DRIVE_ROOT}/data/full/\n"
            "print(f'copy to Drive: {(time.time() - t) / 60:.1f} min')"
        ),
        code(
            "# ---- report these numbers back ----\n"
            "!du -sh {LOCAL_DATA}/shards && ls {LOCAL_DATA}/shards/train-*.tar | wc -l\n"
            "!du -ch {LOCAL_DATA}/shards/train-000[01]*.tar | tail -1   # = 200k subset size\n"
            "!cat {LOCAL_DATA}/pool.stats.json {LOCAL_DATA}/manifests/compose.json\n"
            "!cat {LOCAL_DATA}/shards/*.render.jsonl\n"
            "print(f'generation {GEN_MIN:.1f} min; total notebook {(time.time() - T0) / 60:.1f} min')"
        ),
    ]
    return cells


def nb02() -> list:
    cells = setup_cells(
        "train + evaluate one run",
        "Trains one config from `configs/sweep/` (or `configs/base.yaml` with overrides) with "
        "checkpoints on Drive. **After a disconnect, just run all cells again: training resumes "
        "from the last checkpoint** (model, optimizer, scheduler, RNG and data position).",
    )
    cells += [
        code(
            "RUN_ID = 'A_n200k_f0_s0'     # any row of configs/sweep/index.csv\n"
            "CONFIG = f'configs/sweep/{RUN_ID}.yaml'\n"
            "OUT = f'{DRIVE_ROOT}/runs/{RUN_ID}'\n"
            "OVERRIDES = 'train.ckpt_every=2000 data.num_workers=10'\n"
            "import yaml\n"
            "SIZE = yaml.safe_load(open(CONFIG))['data']['synthetic_max_samples']\n"
            "print(RUN_ID, SIZE)"
        ),
        copy_shards_cell("SIZE // 1000"),
        code(
            "# real-data shards (Am1-A), only needed when real_fraction > 0\n"
            "!mkdir -p /content/real && rsync -a {DRIVE_ROOT}/real/ /content/real/ && ls /content/real"
        ),
        code(
            "t = time.time()\n"
            "!python -m ouroboros.train --config {CONFIG} --out {OUT} --set {OVERRIDES}\n"
            "TRAIN_H = (time.time() - t) / 3600"
        ),
        code(
            "t = time.time()\n"
            "!python scripts/evaluate.py --run {OUT}\n"
            "EVAL_H = (time.time() - t) / 3600\n"
            "print(f'train {TRAIN_H:.2f} GPU-h (this session), eval {EVAL_H:.2f} GPU-h')"
        ),
        code(
            "# ---- report back: the eval printout above, and these lines ----\n"
            "!tail -3 {OUT}/log.jsonl\n"
            "!grep -h img_per_s {OUT}/log.jsonl | tail -1"
        ),
    ]
    return cells


def nb03() -> list:
    cells = setup_cells(
        "3D stage: MACE-OFF benchmarks and error propagation (Phase 6)",
        "Runs the geometry benchmarks on 1,000 test molecules with MACE-OFF23 (GPU) and the "
        "energy-error propagation for one evaluated run (predicted vs. true SMILES, by error "
        "category). Needs `LOCAL_DATA/manifests/test.tsv` (copied from Drive) and a run whose "
        "`eval/predictions.jsonl` exists on Drive.",
    )
    cells += [
        code(
            "!mkdir -p {LOCAL_DATA}/manifests && cp {DRIVE_ROOT}/data/full/manifests/test.tsv {LOCAL_DATA}/manifests/\n"
            "MODEL = 'medium'  # MACE-OFF23 size (Academic Software License)\n"
            "RUN_ID = 'C_n200k_f0_s0'  # the best arm's run (choose after aggregation)\n"
            "N_RELAX = 200  # molecules for the convergence benchmark (10 conformers each)\n"
            "PER_CATEGORY = 100  # error-propagation pairs per error category\n"
            "OUT = f'{DRIVE_ROOT}/results/geometry'\n"
            "os.makedirs(OUT, exist_ok=True)"
        ),
        code(
            "t = time.time()\n"
            "!python scripts/bench_geometry.py embed --manifest {LOCAL_DATA}/manifests/test.tsv --n 1000\n"
            "!python scripts/bench_geometry.py enantio --manifest {LOCAL_DATA}/manifests/test.tsv --n 20 --model {MODEL}\n"
            "!python scripts/bench_geometry.py relax --manifest {LOCAL_DATA}/manifests/test.tsv --n {N_RELAX} --n-conf 10 --model {MODEL}\n"
            "!cp benchmarks/geometry/*.json {OUT}/\n"
            "print(f'benchmarks: {(time.time() - t) / 3600:.2f} GPU-h')"
        ),
        code(
            "t = time.time()\n"
            "!python scripts/error_propagation.py --predictions {DRIVE_ROOT}/runs/{RUN_ID}/eval/predictions.jsonl "
            "--set rendered_test --out {OUT}/{RUN_ID} --n-conf 10 --mmff-prescreen 3 --model {MODEL} "
            "--skip-correct --per-category {PER_CATEGORY}\n"
            "print(f'error propagation: {(time.time() - t) / 3600:.2f} GPU-h')"
        ),
        code(
            "# ---- report back ----\n"
            "!cat {OUT}/embed.json | head -20; cat {OUT}/relax.json; grep -E 'max_abs|pass' {OUT}/enantiomers.json\n"
            "!cat {OUT}/{RUN_ID}/summary.json"
        ),
    ]
    return cells


def write(name: str, cells: list) -> None:
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
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    out = Path(__file__).resolve().parents[1] / "notebooks" / name
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    write("00_colab_setup.ipynb", nb00())
    write("01_generate_data.ipynb", nb01())
    write("02_train_eval.ipynb", nb02())
    write("03_geometry.ipynb", nb03())
