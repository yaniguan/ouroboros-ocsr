#!/bin/bash
#SBATCH --job-name=ouro_geometry
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
export LC_ALL=C LANG=C MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${OURO_ROOT:?}"
cd "$OURO_ROOT/code"
export XDG_CACHE_HOME="$OURO_ROOT/cache"
test -f "$OURO_ROOT/reports/setup.ok"
test -f "$OURO_ROOT/reports/geometry-ready.ok"
test -f "$OURO_ROOT/reports/geometry-cuda.ok"
"$OURO_ROOT/venv/bin/python" scripts/hpc/geometry_batch.py --root "$OURO_ROOT" --task "$SLURM_ARRAY_TASK_ID"
