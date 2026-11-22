# CPU Quantization Check (`lab.quantize`, `PMC val`)

Measured with `python -m lab.quantize` and `scripts/profile_infer.py` on the 963-image `PMC val` split using the single-arm greedy fine-tune v2 model (`A-selected.pt`, same architecture as the target arm-A student).

## Accuracy & throughput (`A_selected_pmc_val_int8.json`, `n = 963`)

| Variant | `PMC val` exact (`n=963`) | Exact (no stereo) | Valid | Mean Tanimoto | Exact drop vs `fp32` | Canonical SMILES agreement |
|---|---|---|---|---|---|---|
| `fp32` | **40.08%** (386/963) | 43.09% | 84.53% | 0.6443 | — | — |
| `int8-dynamic` (decoder `Linear`) | **39.88%** (384/963) | 42.89% | 84.32% | 0.6425 | **-0.21 pp** | **93.46%** |

Note: the timing run shared CPU cores with a background validation job; single-thread CPU-time latency for arm A (`int8` vs `fp32`) is `215 ms` vs `315 ms` p50 (`1.46x` speedup; contended VM wall time `229 ms` vs `376 ms` p50, `1.64x`) in [`benchmarks/cpu-api/README.md`](../cpu-api/README.md). Measurements on the distilled student checkpoint and on a physical Jetson Orin Nano Super remain open in Issue #81.
