#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "ERROR task=%s job=%s array=%s line=%s exit=%s\n" "${OURO_TASK:-unknown}" "${JOB_ID:-local}" "${SGE_TASK_ID:-none}" "$LINENO" "$status" >&2' ERR
export LC_ALL=C LANG=C MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${OURO_ROOT:?}"
: "${OURO_TASK:?}"
cd "$OURO_ROOT/code"
export XDG_CACHE_HOME="$OURO_ROOT/cache"
PY="$OURO_ROOT/venv/bin/python"
test -f "$OURO_ROOT/reports/setup.ok"
DATA="$OURO_ROOT/data/full/shards"
INDEX=${SGE_TASK_ID:-1}
native_gpu() {
  "$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(including_emulation=False); print(torch.cuda.get_device_name())'
}
case "$OURO_TASK" in
  tests)
    "$PY" -m pytest -q
    touch "$OURO_ROOT/reports/tests.ok"
    ;;
  audit)
    "$PY" scripts/hpc/audit_pilot.py --root "$OURO_ROOT"
    ;;
  capacity)
    native_gpu
    test -f "$OURO_ROOT/reports/tests.ok"
    ARMS=(A B C Cplus D)
    ARM=${ARMS[$((INDEX-1))]}
    "$PY" scripts/benchmark_cuda.py --config "configs/sweep/${ARM}_n200k_f0_s0.yaml" \
      --steps 50 --warmup 5 --batch 32 --sequence-length 128 \
      --output "$OURO_ROOT/reports/capacity-${ARM}.json"
    touch "$OURO_ROOT/reports/capacity-${ARM}.ok"
    ;;
  resume)
    native_gpu
    ARMS=(A C)
    ARM=${ARMS[$((INDEX-1))]}
    test -f "$OURO_ROOT/reports/capacity-${ARM}.ok"
    test -f "$OURO_ROOT/reports/data-audit.ok"
    "$PY" scripts/benchmark_cuda_resume.py --config "configs/pilot-${ARM}-200k.yaml" \
      --output "$OURO_ROOT/resume-${ARM}" --checkpoint-step 25 --compared-steps 100 \
      --set "data.synthetic_root=$DATA" "validation.root=$DATA" \
      data.num_workers=4 validation.every=25 validation.samples=32 validation.patience=20
    touch "$OURO_ROOT/reports/resume-${ARM}.ok"
    ;;
  pilot|train)
    native_gpu
    ARMS=(A C)
    ARM=${ARMS[$(((INDEX-1)%2))]}
    SEED=$(((INDEX-1)/2))
    if [ "$OURO_TASK" = train ]; then
      SEED=$((SEED+1))
      test -f "$OURO_ROOT/reports/train-A-s0.ok"
      test -f "$OURO_ROOT/reports/train-C-s0.ok"
    fi
    test -f "$OURO_ROOT/reports/resume-${ARM}.ok"
    test -f "$OURO_ROOT/reports/data-audit.ok"
    OUT="$OURO_ROOT/runs/${ARM}_n200k_f0_s${SEED}"
    "$PY" -m ouroboros.train --config "configs/pilot-${ARM}-200k.yaml" --out "$OUT" \
      --set "seed=$SEED" "data.synthetic_root=$DATA" "validation.root=$DATA" \
      "eval.sets.rendered_test.root=$DATA" data.num_workers=4 train.device=cuda eval.batch_size=32
    touch "$OURO_ROOT/reports/train-${ARM}-s${SEED}.ok"
    ;;
  evaluate)
    native_gpu
    ARMS=(A C)
    ARM=${ARMS[$(((INDEX-1)%2))]}
    SEED=$(((INDEX-1)/2))
    test -f "$OURO_ROOT/reports/train-${ARM}-s${SEED}.ok"
    "$PY" scripts/evaluate.py --run "$OURO_ROOT/runs/${ARM}_n200k_f0_s${SEED}"
    touch "$OURO_ROOT/reports/eval-${ARM}-s${SEED}.ok"
    ;;
  *) echo "Unknown task $OURO_TASK" >&2; exit 2 ;;
esac
