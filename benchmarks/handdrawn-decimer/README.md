# DECIMER Hand-drawn evaluation subset (`handdrawn-decimer`)

A seeded 200-image hand-drawn evaluation set sampled from the public **DECIMER Hand-drawn molecule images dataset** (Brinkhaus et al., *J. Cheminform.* 14:36, 2022; Zenodo record `6456306`, DOI `10.5281/zenodo.6456306`, CC-BY 4.0, 5,088 depictions). Only the image-free manifest (`manifest.csv`), leakage receipt (`leakage.json`), and evaluation reports are committed here; images stay out of git ([`DATA.md`](../../DATA.md)).

## Identity and counts

| Field | Value |
|---|---|
| Upstream dataset | DECIMER Hand-drawn (`10.5281/zenodo.6456306`, CC-BY 4.0) |
| Seed | `2026` (`HANDDRAWN_DECIMER_SEED`) |
| Rows | `200` (`DECIMER_HDM` 200; all 200 standardize cleanly in RDKit) |
| Stereo rows (`has_stereo`) | `46` (`no_stereo`: `154`) |
| Unique connectivity blocks | `200` |
| `manifest.csv` SHA-256 | `076681774e739e10c289d8e50d31ce7980bd9a20bf4acab45a710d623f53d6de` |

Pinned in [`ouroboros/data/real.py`](../../ouroboros/data/real.py) as `EXPECTED_HANDDRAWN_DECIMER_MANIFEST_SHA256`.

## Zero overlap (`leakage.json`)

Checked with `scripts/check_leakage.py --fail-on-connectivity` (both full standard InChIKey and 14-character connectivity block):

- **0 overlap** with `subset100` (`295` unique keys)
- **0 overlap** with `frozen-test-v2` (`300` unique keys)
- **0 overlap** with `pmc_train` (`13,133` unique keys) and `pmc_val` (`847` unique keys)
- **0 overlap** with `synth_train` (`50,000` synthetic fine-tuning molecules)

## Results (200 hand-drawn images, zero hand-drawn training data)

| Model | All (`n=200`) | 95% CI | Exact (no stereo) | `has_stereo` (`n=46`) | `no_stereo` (`n=154`) | Mean Tanimoto | Report |
|---|---|---|---|---|---|---|---|
| **A + C ensemble (v2), beam 5 + `pad` TTA** | **4.50%** | `[2.00%, 7.50%]` | 6.00% | 4.35% | 4.55% | 0.3116 | [`AC-ensemble-beam5-ttapad-report.md`](AC-ensemble-beam5-ttapad-report.md) |
| **A + C ensemble (v2), beam 5** | **4.50%** | `[2.00%, 7.50%]` | 6.50% | 4.35% | 4.55% | 0.2980 | [`AC-ensemble-beam5-report.md`](AC-ensemble-beam5-report.md) |
| A fine-tuned v2, greedy | 3.00% | `[1.00%, 5.50%]` | 5.00% | 2.17% | 3.25% | 0.2383 | [`A-selected-report.md`](A-selected-report.md) |
| C fine-tuned v2, greedy | 2.00% | `[0.50%, 4.50%]` | 3.50% | 2.17% | 1.95% | 0.2439 | [`C-selected-report.md`](C-selected-report.md) |
| Released A (`v0.1.0-alpha`), greedy | 0.00% | `[0.00%, 0.00%]` | 0.00% | 0.00% | 0.00% | 0.1293 | [`A-released-report.md`](A-released-report.md) |
| Released B (`v0.1.0-alpha`), greedy | 0.00% | `[0.00%, 0.00%]` | 0.00% | 0.00% | 0.00% | 0.1304 | [`B-released-report.md`](B-released-report.md) |
| Released C (`v0.1.0-alpha`), greedy | 0.00% | `[0.00%, 0.00%]` | 0.00% | 0.00% | 0.00% | 0.1372 | [`C-released-report.md`](C-released-report.md) |

Neither `v0.1.0-alpha` nor fine-tune v2 trained on hand-drawn images (`ouroboros.data.augment.handdrawn_augment` was added after v2 training). Published full-set numbers on DECIMER Hand-drawn (5,088 images or its HD-test split) are not directly comparable to this 200-image sample until scoring parity is checked.

## Materializing images locally

```bash
python scripts/fetch_handdrawn_subset.py \
    --from-manifest benchmarks/handdrawn-decimer/manifest.csv \
    --out real_data/handdrawn_decimer
python scripts/ingest_real.py \
    --manifest real_data/handdrawn_decimer/manifest.csv \
    --out real/handdrawn_decimer
```
