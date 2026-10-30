# v0.1.0-alpha checkpoints on a small real-document subset (2026-10-29)

**Data:** `Keylab/OCSR-Benchmarks` (Hugging Face), 100 images each from USPTO, UOB and ACS,
seed 0 (`scripts/fetch_real_subset.py`). Ingested with `scripts/ingest_real.py` (384 px):
278 of 300 kept; 22 labels failed to parse (R-groups/abbreviations such as `[R]`, mostly ACS,
plus 2 USPTO). Images and predictions are not committed (DATA.md).

**Models:** released `ouroboros-{A,B,C}.pt`, greedy decoding, angle 0, CPU (8 cores, shared with
another job; 660–1017 s per arm for 278 images). Scored with `scripts/eval_predictions.py`.

| arm | n | exact (95% CI) | exact no-stereo | InChIKey | valid | mean Tanimoto |
|---|---|---|---|---|---|---|
| A (CNN) | 278 | 19.42% [14.75, 24.10] | 20.50% | 21.22% | 80.58% | 0.4528 |
| B (CNN + rot. aug.) | 278 | 17.99% [13.31, 22.30] | 17.99% | 19.42% | 76.98% | 0.4239 |
| C (C8 steerable) | 278 | 17.99% [13.67, 22.30] | 17.99% | 19.42% | 80.94% | 0.4507 |

Per subset, exact match A / B / C: UOB 41 / 39 / 39 % (n=100); USPTO 13.3 / 11.2 / 10.2 % (n=98);
ACS 0 / 0 / 1.25 % (n=80). Full tables: `{A,B,C}_report.md`.

These are the first real-document numbers for the synthetic-only pilot; differences between arms
are within the CIs. Scoring parity with arXiv:2608.09100 remains unverified.

## Reproduce

```bash
python -m ouroboros.weights fetch --arm all
python scripts/fetch_real_subset.py --sets USPTO UOB ACS --n 100 --out real_data/subset100
python scripts/ingest_real.py --manifest real_data/subset100/manifest.csv --out real/subset100
python scripts/predict_real.py --ckpt checkpoints/ouroboros-C.pt --shards real/subset100 --out runs/real_subset100/C_preds.csv
python scripts/eval_predictions.py runs/real_subset100/C_preds.csv --json runs/real_subset100/C_report.json --md runs/real_subset100/C_report.md
```
