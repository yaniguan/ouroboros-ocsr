#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C
export OURO_ROOT=/u/scratch/y/yaniguan/ouroboros-20260930
V0_ROOT=/u/scratch/y/yaniguan/ouroboros-v0-20261001
source /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
export PATH="$OURO_ROOT/container-bin:$PATH"
export APPTAINER_CACHEDIR="$OURO_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-$OURO_ROOT/cache/apptainer}"
task_gpu=${CUDA_VISIBLE_DEVICES:-${SGE_HGR_cuda:-}}
test -n "$task_gpu"
task_gpu=${task_gpu// /,}
image=$(cat "$OURO_ROOT/reports/container-image.txt")
apptainer exec --cleanenv --nv --bind "$OURO_ROOT:$OURO_ROOT" --bind "$V0_ROOT:$V0_ROOT" \
 --env "CUDA_VISIBLE_DEVICES=$task_gpu,JOB_ID=$JOB_ID,MPLBACKEND=Agg,OMP_NUM_THREADS=4" \
 "$image" "$OURO_ROOT/venv/bin/python" -u "$V0_ROOT/v0_baseline_b.py" \
 --original "$OURO_ROOT" --root "$V0_ROOT"
