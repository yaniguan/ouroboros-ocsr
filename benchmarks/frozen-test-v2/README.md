# Frozen real test set v2 (`frozen-test-v2`)

An untouched 300-image real test set drawn from `Keylab/OCSR-Benchmarks` for future release evaluations. Only the image-free manifest (`manifest.csv`) and leakage receipt (`leakage.json`) are committed here; images stay out of git ([`DATA.md`](../../DATA.md)).

## Rule: score once per release

The earlier 298-image `subset100` sample (`seed=0`) was watched across many rounds of model and decoding work, so it is now a **dev-test** set.

`frozen-test-v2` must stay untouched during day-to-day work:

1. Do all model, checkpoint, hyperparameter, and decoding selection on validation (`PMC val`), or sanity checks on `subset100` (`dev-test`).
2. Score `frozen-test-v2` **at most once per release**, after all training and decoding choices are fixed.
3. Include `benchmarks/frozen-test-v2/manifest.csv` in `scripts/check_leakage.py --dump-eval-keys` whenever building new training or validation splits.

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
