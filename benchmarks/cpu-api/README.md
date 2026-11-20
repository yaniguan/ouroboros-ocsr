# CPU inference API (`ouroboros/infer.py`): latency and accuracy

Produced by `scripts/bench_infer.py`; raw output in [`bench.json`](bench.json).

## Setup

- CPU: `Intel(R) Xeon(R) Processor` (8 logical CPUs), 1 torch thread, torch 2.10 (CPU).
- The machine was shared with other jobs (load average 12–19). **Wall** time includes waiting for the CPU. **CPU** time is the process time of the request, i.e. its cost on an idle core.
- 298 real images from `real_data/subset100/manifest.csv` (ACS, UOB and USPTO; rows whose gold SMILES does not standardize are skipped). Each request starts from the raw file bytes and covers decode, resize, encode, greedy decode, rescoring and RDKit standardization.
- Weights: release `v0.1.0-alpha` arms A and C, trained on synthetic images only. The fine-tuned v2 checkpoints (51–54% on this set with ensembling) were not available here.

## Results

| Variant | wall p50 / p95 (ms) | CPU p50 / p95 (ms) | exact | valid | ACS / UOB / USPTO exact |
|---|---|---|---|---|---|
| **A, int8 decoder (default)** | 229 / 359 | 215 / 331 | 18.8% | 78.9% | 0% / 41% / 15.3% |
| A, fp32 | 376 / 1145 | 315 / 560 | 18.1% | 78.2% | 0% / 41% / 13.3% |
| C (C8 steerable, exported), fp32 | 552 / 1406 | 489 / 659 | 16.8% | 79.2% | 1% / 39% / 10.2% |

- int8 matches fp32 accuracy on this set; 89.6% of its SMILES are identical to fp32's.
- The fp32 wall p95 above 1 s comes from contention on the shared machine. Its CPU time stays under 1 s.
- A fp32 greedy reproduces the earlier subset100 result for this arm: 19.4% on the 278 images ingested at the time, see `benchmarks/real-subset100`.

## Confidence flag

`confidence` is the mean per-token probability of the decoded SMILES. `low_confidence` is set below the threshold or when RDKit rejects the SMILES. The threshold was picked on the even rows and checked on the odd rows:

Default variant (A, int8). Each half has 149 images.

| threshold | even rows: coverage / exact among accepted | odd rows: coverage / exact among accepted |
|---|---|---|
| 0.90 | 58% / 30% | 53% / 38% |
| 0.95 | 40% / 43% | 41% / 46% |
| **0.98 (default)** | 20% / 63% | 21% / 72% |

At 0.98, A fp32 gives 23% coverage and 69% exact on the odd rows. Arm C does worse there: 22% coverage and 45% exact.

Accepted predictions are much more often right than the 18% base rate, but the flag is not a calibrated probability: at 0.98 only about 32 images per half are accepted, and all images come from three document sources.

## Reproduce

```bash
python scripts/bench_infer.py --manifest real_data/subset100/manifest.csv --out benchmarks/cpu-api/bench.json
```
