# Rotated Real Document Figures (`UOB` dev-test, `n = 100`)

Direct comparison of released `v0.1.0-alpha` arms on **real scanned chemical figures**
from `Keylab/OCSR-Benchmarks` (`UOB`, `seed=0`, `n=100` dev-test) rotated across 8
orientations (`0°..315°` in `45°` steps; `int8` dynamic-quantized greedy decoding).

| Arm | Encoder & training | `0°` upright | `90°/180°/270°` mean | Off-grid (`45°..315°`) mean | 8-angle mean |
|---|---|---:|---:|---:|---:|
| **`A`** | Plain CNN (`17.8M` enc, 0 rotated train imgs) | 41.0% | **10.0%** | 2.5% | **10.1%** |
| **`B`** | Plain CNN + 360° rot aug (`17.8M` enc) | 39.0% | **33.3%** | 39.2% | **37.0%** |
| **`C_pure`** | Pure-PyTorch `C8` steerable (`7.0M` enc, **0 rotated train imgs**, zero `escnn`) | 39.0% | **39.0%** | 36.0% | **37.5%** |

## Per-angle exact match (`n = 100` real `UOB` figures)

| Arm | `0°` | `45°` | `90°` | `135°` | `180°` | `225°` | `270°` | `315°` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`A`** | 41.0% | 4.0% | 1.0% | 0.0% | 28.0% | 4.0% | 1.0% | 2.0% |
| **`B`** | 39.0% | 36.0% | 34.0% | 47.0% | 34.0% | 35.0% | 32.0% | 39.0% |
| **`C_pure`** | 39.0% | 36.0% | 39.0% | 36.0% | 39.0% | 36.0% | 39.0% | 36.0% |

## Takeaway

- On upright (`0°`) real figures, plain CNN `A` holds a small edge because document scans
  have consistent horizontal text/bond layout and `A` has `2.5x` more encoder parameters.
- As soon as real scanned figures are rotated (`90°`, `180°`, `270°` — common in landscape
  patent tables and rotated page scans — or off-grid `45°` steps), plain CNN `A` collapses,
  whereas pure-PyTorch `C8` steerable `C_pure` retains its accuracy **without a single
  rotated training image** and with zero `escnn` / `gfortran` dependency at inference.

## Reproduce

```bash
python scripts/fetch_real_subset.py --sets UOB --n 100 --seed 0 --out real_data/subset100_uob
python scripts/ingest_real.py --manifest real_data/subset100_uob/manifest.csv --out real/subset100_uob
python scripts/bench_rotated_real.py --shards real/subset100_uob \
    --out benchmarks/rotated-real-uob/summary.json --md benchmarks/rotated-real-uob/README.md
```
