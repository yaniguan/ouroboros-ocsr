# Selective Confidence Gate (`PMC val` selection & `frozen-test-v2` check)

Thresholds chosen strictly on `PMC val` (`963` images) with `scripts/predict_real.py --nbest-out` and `scripts/eval_selective.py` (`AC-ensemble-beam5-ttapad` teacher), recorded in [`lab/configs/edge_distill.json`](../../lab/configs/edge_distill.json), and then evaluated **once** on `frozen-test-v2` (`300` images).

## 1. Validation selection (`PMC val`, `n = 963`, ungated exact = `44.55%`)

Full grid in [`pmc_val_grid.json`](pmc_val_grid.json); `eval_selective.py` outputs in [`pmc_val_disagreement_0.0.json`](pmc_val_disagreement_0.0.json) and [`pmc_val_disagreement_0.25.json`](pmc_val_disagreement_0.25.json).

| `max_disagreement` | `min_margin` | Accepted (`n=963`) | Coverage | Exact precision (with stereo) |
|---|---|---|---|---|
| `none` | `0.05` | 484 | 50.26% | 69.42% |
| `0.50` | `0.05` | 391 | 40.60% | 81.33% |
| `0.25` | `0.05` | 272 | 28.25% | 87.87% |
| **`0.00` (locked)** | **`0.05` (locked)** | **192** | **19.94%** | **92.19%** |
| `0.00` | `0.10` | 166 | 17.24% | 93.98% |
| `0.00` | `0.20` | 70 | 7.27% | 95.71% |

Locked thresholds for `lab/configs/edge_distill.json`: `min_margin = 0.05`, `max_disagreement = 0.0` (all four `(model, view)` sources agree on the standardized molecule and the top-2 molecule score gap is at least `0.05`).

## 2. Single-shot held-out check (`frozen-test-v2`, `n = 300`)

Summary in [`frozen_test_v2_gated_summary.json`](frozen_test_v2_gated_summary.json) (underlying SMILES predictions match [`benchmarks/frozen-test-v2/AC-ensemble-beam5-ttapad-preds.csv`](../frozen-test-v2/AC-ensemble-beam5-ttapad-preds.csv) row for row).

| Subset | `n` | Ungated exact | Accepted | Coverage | Accepted exact | Selective precision |
|---|---|---|---|---|---|---|
| **ALL** | 300 | 46.67% | **90** | **30.00%** | **88** | **97.78%** |
| ACS | 100 | 30.00% | 10 | 10.00% | 10 | **100.00%** |
| UOB | 100 | 85.00% | 70 | 70.00% | 68 | **97.14%** |
| USPTO | 100 | 25.00% | 10 | 10.00% | 10 | **100.00%** |

Passing the gated CSV through `lab/molecule_context.py` yields `85` `status=ok` rows (`28.33%` coverage, **`97.65%` exact precision**, `83/85`), `5` `review` rows (`pseudo_atoms` / `unassigned_stereo`), `209` `rejected` rows, and `1` `invalid` row.

## 3. Coverage skew & CPU latency cost (`operating_modes.json`)

- **Coverage is concentrated on clean achiral molecules:** `70/90` accepted predictions (`77.8%` of accepted items) come from `UOB` (`70.0%` subset coverage), whereas harder `ACS` journal figures (`R`-groups/abbreviations) and `USPTO` patent structures (long sequences + stereo) each have **`10.0%` coverage** (`90%` routed to human review).
- **Compute trade-off vs. single-arm `int8` greedy API:** the `97.78%` selective gate runs 2 models × 2 TTA views (`4` encoder passes + `20` beams + cross-rescoring), which takes **`~9.5x` the single-thread CPU wall time** of single-arm `A` `int8` greedy (`4,074 ms` vs `429 ms` p50 on local Windows CPU; see [`benchmarks/cpu-api/operating_modes.json`](../cpu-api/operating_modes.json)). Distilling this ensemble teacher into a single greedy student is tracked in [`lab/edge_plan.py`](../../lab/edge_plan.py) (`#80`).
