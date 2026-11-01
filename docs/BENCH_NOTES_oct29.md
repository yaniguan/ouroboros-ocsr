# Bench notes — 2026-10-29 (branch `feat/bench-oct29`)

## Status

Run on 2026-10-29 (CPU, venv311). Full suite: **206 passed** (was 163). No real-benchmark predictions or USPTO/UOB/DECIMER images exist locally, so the harness was run only on the synthetic sanity set below. Those numbers check the scorer itself; they are not model accuracy.

## What exists already (reused, not duplicated)

`ouroboros/eval/standardize.py` and `ouroboros/eval/metrics.py` already provided RDKit
standardization (ChargeParent), stereo-aware exact match (`exact`), stereo-agnostic match
(`exact_flat`), standard InChIKey match, Morgan r=2 / 2048-bit Tanimoto, per-stereo-element
accuracy, `aggregate` and `bootstrap_ci`. Per-set reporting rules (Am1-D: rendered and real
sets are never pooled) live in `evaluate.py` / `aggregate.py`.

## What was added

| File | Purpose |
|---|---|
| `ouroboros/eval/metrics.py` | `aggregate` adds `tanimoto_1` (Tanimoto@1.0). It is derived from the stored `tanimoto` field, so archived `predictions.jsonl` logs still summarize. `PairScore` is unchanged. |
| `ouroboros/eval/smiles.py` | String-level API on top of the same rules: `canonicalize(stereo=, standardize=)`, `is_valid`, `inchikey(connectivity_only=)`, `exact_match(stereo=)`, `inchikey_match`, `tanimoto(radius=, n_bits=, chirality=)`, `score_batch`, `batch_metrics` (counts invalid golds instead of raising). With default arguments each one agrees with the matching `score_pair` field. |
| `ouroboros/eval/harness.py`, `scripts/eval_predictions.py` | Predictions-file CLI. Reads csv/tsv/jsonl with column aliases (`id/key/index`, `gold/ref/reference/smiles`, `pred/prediction`, `subset/source/set`, `kind`), takes `--where COL=VALUE` filters and accepts the `run_eval` `predictions.jsonl` as is. Reports n, exact with a 95% bootstrap CI, exact without stereo, InChIKey, valid %, mean Tanimoto and Tanimoto@1.0, overall and per subset. Writes `--json`, `--md` and `--per-sample` output; `--workers` sets parallelism. |
| `ouroboros/eval/sanity.py`, `scripts/make_sanity_preds.py` | A model-free sanity set: 24 gold molecules × 3 subsets (`render`, `mirror`, `perturb`) with documented expected outcomes. The `render` and `mirror` predictions are the labels RDKit re-perceives from drawings made by `ouroboros.data.render`. |
| `tests/test_smiles_utils.py`, `tests/test_harness.py`, `tests/test_sanity_set.py`, `tests/test_metrics.py` (+1 assertion) | Tests (all pass). |

Harness rules:

* A blank or missing prediction counts as invalid.
* A gold that does not standardize is excluded from scoring, but it is counted and its id is listed.
* A duplicate (subset, kind, id) is an error.
* With several `kind`s there is one overall row per kind and no cross-kind number.
* The parity footnote is attached while `PARITY_VERIFIED` is False.
* Tanimoto ignores chirality by default, so an enantiomer still counts toward Tanimoto@1.0.

## Data available locally

There are no real eval sets, run directories, checkpoints or per-sample predictions in the
checkout (`data/`, `real/`, `runs/` and `checkpoints/` are absent). The archived
`benchmarks/v0.1.0-alpha/*-summary.json` files hold only aggregates. The release weights were
not downloaded (that is out of scope for a CPU-only, no-download session). The only available
input is therefore the synthetic sanity set.

## Commands to run (in order)

```bash
source /workspace/venv311/bin/activate
cd /workspace/repos/ouroboros-ocsr
python -m pytest -q tests/test_smiles_utils.py tests/test_harness.py \
    tests/test_sanity_set.py tests/test_metrics.py
python -m pytest -q                      # full suite
ruff check ouroboros scripts tests       # if ruff is installed
python scripts/make_sanity_preds.py --out benchmarks/eval-harness/sanity_preds.csv
python scripts/eval_predictions.py benchmarks/eval-harness/sanity_preds.csv \
    --json benchmarks/eval-harness/sanity_report.json \
    --md benchmarks/eval-harness/sanity_report.md \
    --per-sample /tmp/sanity_per_sample.jsonl
```

Expected results on the sanity set, before running (these are expectations, not measurements):

* `render`: exact 100%, unless the renderer mis-depicts some stereo. Any miss is a real finding
  about the label path (the dataset builder drops such rows as `label_mismatch`).
* `mirror`: exact-without-stereo 100%. Exact match only for molecules with no tetrahedral
  stereo or that are achiral (meso / ring cis–trans).
* `perturb`: 4 per perturbation type. `random_smiles` and `salt` → exact. `no_stereo` →
  exact only when the gold has no stereo. `add_methyl` → valid but wrong. `syntax_error` and
  `blank` → invalid. Valid % = 16/24.

## Measured numbers

`make_sanity_preds.py` → 72 rows (24 render-faithful, 24 mirror-image stereo flips, 24 perturbed/invalid). `eval_predictions.py` output:

| subset | n | exact | exact 95% CI | exact (no stereo) | InChIKey | valid | mean Tanimoto | Tanimoto@1.0 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 72 | 62.50% | [51.39%, 73.61%] | 83.33% | 62.50% | 88.89% | 0.8551 | 83.33% |
| mirror | 24 | 45.83% | [29.17%, 66.67%] | 100.00% | 45.83% | 100.00% | 1.0000 | 100.00% |
| perturb | 24 | 41.67% | [20.83%, 62.50%] | 50.00% | 41.67% | 66.67% | 0.5654 | 50.00% |
| render | 24 | 100.00% | [100.00%, 100.00%] | 100.00% | 100.00% | 100.00% | 1.0000 | 100.00% |

Expected behaviour holds: render subset 100%; mirror flips are caught by stereo-aware exact/InChIKey but invisible to stereo-agnostic exact and Tanimoto; invalid perturbations lower validity and Tanimoto.
