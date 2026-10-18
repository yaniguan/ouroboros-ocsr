#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${OURO_ROOT:?}"
cd "$OURO_ROOT/code"
test -f "$OURO_ROOT/reports/geometry-cuda.ok"
"$OURO_ROOT/venv/bin/python" -c 'import torch; assert torch.cuda.is_available()'
"$OURO_ROOT/venv/bin/python" scripts/bench_geometry.py enantio \
  --manifest "$OURO_ROOT/data/full/manifests/test.tsv" --n 20 \
  --model "$OURO_ROOT/models/MACE-OFF23_medium.model"
