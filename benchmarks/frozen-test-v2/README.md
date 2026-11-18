# Frozen real test set v2 (`frozen-test-v2`)

An untouched 300-image real test set drawn from `Keylab/OCSR-Benchmarks` for release evaluation. Only the image-free manifest (`manifest.csv`), leakage receipt (`leakage.json`), `PMC val` decoding receipt (`val_selection.json`), and single-shot release evaluation files (`AC-ensemble-beam5-ttapad-*`) are committed here; images stay out of git ([`DATA.md`](../../DATA.md)).

## Rule: score once per release

The earlier 298-image `subset100` sample (`seed=0`) was watched across many rounds of model and decoding work, so it is kept as a **dev-test** set (`54.0%` exact match).

`frozen-test-v2` stays untouched during day-to-day work:

1. Do all model, checkpoint, hyperparameter, and decoding selection on validation (`PMC val`), or sanity checks on `subset100` (`dev-test`).
2. Score `frozen-test-v2` **at most once per release**, after all training and decoding choices are fixed on validation.
3. Include `benchmarks/frozen-test-v2/manifest.csv` in `scripts/check_leakage.py --dump-eval-keys` whenever building new training or validation splits.

## Fine-tune v2 single-shot evaluation

### 1. Decoding locked on `PMC val` (`val_selection.json`, `n = 963`)

All ten candidate decoding variants from `colab/finetune_v2.ipynb` were scored on the 963-image `PMC val` split alone (`subset100` and `frozen-test-v2` were not consulted):

| Variant | `PMC val` exact (`n=963`) | Exact (no stereo) | Valid |
|---|---|---|---|
| **`AC-ensemble-beam5-ttapad` (locked)** | **44.55%** (`[41.43%, 47.66%]`) | 48.81% | 99.07% |
| `A-selected-beam5-ttapad` | 42.37% | 46.21% | 96.68% |
| `AC-ensemble-beam5` | 42.26% | 46.83% | 97.61% |
| `A-selected` (greedy) | 40.08% | 43.09% | 84.53% |
| `A-selected-beam5` | 39.98% | 43.41% | 94.29% |
| `C-selected-beam5-ttapad` | 33.13% | 35.62% | 97.51% |
| `C-selected-beam5` | 31.57% | 34.16% | 94.91% |
| `C-selected` (greedy) | 31.05% | 32.92% | 85.15% |
| `A-released` (`v0.1.0-alpha`) | 7.89% | 8.31% | 57.01% |
| `C-released` (`v0.1.0-alpha`) | 7.37% | 7.58% | 61.37% |

### 2. Single-shot `frozen-test-v2` score (`AC-ensemble-beam5-ttapad-report.md`, `n = 300`)

Scored once with `scripts/predict_real.py --ckpt A-selected.pt --ckpt C-selected.pt --beam 5 --tta pad` and `scripts/eval_predictions.py`:

| Subset | `n` | Exact | 95% CI | Exact (no stereo) | InChIKey | Valid | Mean Tanimoto |
|---|---|---|---|---|---|---|---|
| **ALL** | 300 | **46.67%** | `[40.67%, 51.67%]` | 49.33% | 49.67% | 99.67% | 0.7609 |
| ACS | 100 | 30.00% | `[21.00%, 39.00%]` | 35.00% | 30.00% | 100.00% | 0.6388 |
| UOB | 100 | 85.00% | `[78.00%, 91.00%]` | 87.00% | 92.00% | 100.00% | 0.9437 |
| USPTO | 100 | 25.00% | `[16.00%, 33.00%]` | 26.00% | 27.00% | 99.00% | 0.7001 |
| `has_stereo` | 57 | 1.75% | `[0.00%, 5.26%]` | 14.04% | 1.75% | 98.25% | 0.5516 |
| `no_stereo` | 243 | 57.20% | `[51.03%, 63.37%]` | 57.61% | 60.91% | 100.00% | 0.8100 |

Do not run additional fine-tune v2 decoding variants on this set.

## Identity and counts

| Field | Value |
|---|---|
| Upstream dataset | `Keylab/OCSR-Benchmarks` (`2fdcdc59067acb36d84ddc82c1ace24c91d5ad71`) |
| Seed | `2026` (`FROZEN_TEST_V2_SEED`) |
| Rows | `300` (`USPTO` 100, `UOB` 100, `ACS` 100; all 300 parse and standardize cleanly) |
| Stereo rows (`has_stereo`) | `57` (`USPTO` 33, `UOB` 1, `ACS` 23) |
| Unique connectivity blocks | `300` |
| `manifest.csv` SHA-256 | `e4a24c68b3d893498b0df727bdb5b77eed0b98e66f6898f21d17aecf01ee5e1a` |

Pinned in [`ouroboros/data/real.py`](../../ouroboros/data/real.py) as `EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256`.

## Verified zero overlap (`leakage.json`)

Checked with `scripts/check_leakage.py --fail-on-connectivity` (full InChIKey and 14-character connectivity block, so stereoisomers also count as overlap):

- **0 overlap** with `subset100` (`seed=0` dev-test set, 300 rows / 295 unique molecules)
- **0 overlap** with `yaniguan/ocsr-dataset` (`8e00509a5e70c42f05693c4d7268fcdb08d50711`) `train` (17,249 rows / 13,151 unique keys) and `val` (1,193 rows / 1,015 unique keys)
- **0 overlap** with `zinc250k` (249,432 unique keys)

## Materializing images locally (for a release check)

```bash
python scripts/fetch_real_subset.py \
    --from-manifest benchmarks/frozen-test-v2/manifest.csv \
    --out real_data/frozen_test_v2
python scripts/ingest_real.py \
    --manifest real_data/frozen_test_v2/manifest.csv \
    --out real/frozen_test_v2
```

`--from-manifest` checks `manifest.csv` against `EXPECTED_FROZEN_TEST_V2_MANIFEST_SHA256`, downloads the pinned `Keylab/OCSR-Benchmarks` parquets, writes `real_data/frozen_test_v2/images/<SET>/<id>.png`, and copies `manifest.csv` unchanged.
