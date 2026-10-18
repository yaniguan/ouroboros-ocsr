#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "ERROR setup job=%s line=%s exit=%s\n" "${JOB_ID:-${SLURM_JOB_ID:-local}}" "$LINENO" "$status" >&2' ERR
export LC_ALL=C LANG=C MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${OURO_ROOT:?}"
cd "$OURO_ROOT/code"
mkdir -p "$OURO_ROOT/cache" "$OURO_ROOT/reports" "$OURO_ROOT/bootstrap"
export UV_CACHE_DIR="$OURO_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$OURO_ROOT/python"
export PIP_CACHE_DIR="$OURO_ROOT/cache/pip"
export XDG_CACHE_HOME="$OURO_ROOT/cache"
hostname
ldd --version
command -v gfortran
UV="$OURO_ROOT/bootstrap/bin/uv"
if [ ! -x "$UV" ]; then
  python3 -m pip install --disable-pip-version-check --target "$OURO_ROOT/bootstrap" uv
fi
STAGED_PY="$OURO_ROOT/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
if [ -x "$STAGED_PY" ]; then
  "$UV" venv --python "$STAGED_PY" "$OURO_ROOT/venv"
elif command -v python3.11 >/dev/null 2>&1 && python3.11 -c 'import pathlib,sysconfig; assert (pathlib.Path(sysconfig.get_path("include"))/"Python.h").is_file()'; then
  "$UV" venv --python "$(command -v python3.11)" "$OURO_ROOT/venv"
else
  "$UV" venv --python 3.11 --python-preference only-managed "$OURO_ROOT/venv"
fi
PY="$OURO_ROOT/venv/bin/python"
"$UV" pip install --python "$PY" -r requirements.txt
"$UV" pip install --python "$PY" --no-deps -e .
"$UV" pip check --python "$PY"
"$UV" pip freeze --python "$PY" > "$OURO_ROOT/reports/environment.txt"
"$PY" -c 'import torch, escnn, mace, rdkit; print(torch.__version__, rdkit.__version__)'
touch "$OURO_ROOT/reports/setup.ok"
