#!/bin/bash
set -euo pipefail
export LC_ALL=C LANG=C MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${OURO_ROOT:?}"
cd "$OURO_ROOT/code"
test -f "$OURO_ROOT/reports/geometry-ready.ok"
"$OURO_ROOT/venv/bin/python" - <<'PY'
import os
from pathlib import Path
import torch
from ouroboros.geometry.conformers import lowest_energy_conformer, mace_calculator
from ouroboros.provenance import atomic_json
assert torch.cuda.is_available()
root = Path(os.environ['OURO_ROOT'])
calc = mace_calculator(str(root / 'models/MACE-OFF23_medium.model'), device='cuda', dtype='float64')
result = lowest_energy_conformer('C[C@H](N)C(=O)O', n_conf=1, seed=0, calc=calc)
report = {'gpu': torch.cuda.get_device_name(), 'job_id': os.environ['SLURM_JOB_ID'],
          'status': result.status, 'converged': result.converged, 'seconds': result.seconds}
atomic_json(root / 'reports/geometry-cuda-probe.json', report)
print(report)
assert result.status == 'ok' and result.converged and result.stereo_ok_final
(root / 'reports/geometry-cuda.ok').touch()
PY
