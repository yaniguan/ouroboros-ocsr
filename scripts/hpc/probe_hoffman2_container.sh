#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "ERROR container CUDA probe line=%s exit=%s\n" "$LINENO" "$status" >&2' ERR
export LC_ALL=C LANG=C
: "${OURO_ROOT:?}"
test -f "$OURO_ROOT/reports/container-setup.ok"
source /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
export PATH="$OURO_ROOT/container-bin:$PATH"
export APPTAINER_CACHEDIR="$OURO_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-$OURO_ROOT/cache/apptainer}"
task_gpu=${CUDA_VISIBLE_DEVICES:-${SGE_HGR_cuda:-}}
printf 'CUDA_VISIBLE_DEVICES=%s SGE_HGR_cuda=%s\n' "${CUDA_VISIBLE_DEVICES:-}" "${SGE_HGR_cuda:-}"
test -n "$task_gpu" || { echo 'No scheduler GPU binding; refusing an unbound CUDA probe' >&2; exit 1; }
task_gpu=${task_gpu// /,}
image=$(cat "$OURO_ROOT/reports/container-image.txt")
cd "$OURO_ROOT/code"
apptainer exec --cleanenv --nv --bind "$OURO_ROOT:$OURO_ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$task_gpu,MPLBACKEND=Agg,OMP_NUM_THREADS=4" \
  "$image" "$OURO_ROOT/venv/bin/python" scripts/benchmark_cuda.py \
  --config configs/pilot-A-200k.yaml --batch 2 --sequence-length 32 --steps 5 --warmup 1 \
  --output "$OURO_ROOT/reports/container-cuda-probe.json"
touch "$OURO_ROOT/reports/container-cuda.ok"
