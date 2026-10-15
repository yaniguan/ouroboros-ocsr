#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${OURO_ROOT:?}"
cd "$OURO_ROOT/code"
test -f "$OURO_ROOT/reports/setup.ok"
mkdir -p "$OURO_ROOT/models"
curl --fail --location --retry 3 --max-time 300 \
  https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model \
  -o "$OURO_ROOT/models/MACE-OFF23_medium.model.part"
mv "$OURO_ROOT/models/MACE-OFF23_medium.model.part" "$OURO_ROOT/models/MACE-OFF23_medium.model"
sha256sum "$OURO_ROOT/models/MACE-OFF23_medium.model" "$OURO_ROOT/data/full/manifests/test.tsv" > "$OURO_ROOT/reports/geometry-inputs.sha256"
"$OURO_ROOT/venv/bin/python" -m pytest -q tests/test_geometry.py
touch "$OURO_ROOT/reports/geometry-ready.ok"
