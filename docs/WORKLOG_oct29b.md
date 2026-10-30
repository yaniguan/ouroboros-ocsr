# Oct 29: labelled R-groups, real-style synthetic augmentation and CPU evaluation

Implemented in the supplied `feat/rgroups-aug-oct29` workspace. No git commands were run and no commit was made. Project interpreter: `/workspace/venv311/bin/python`. All inference used CPU, two OMP/MKL/OpenBLAS threads, batch 8; tests/augmentation benchmarks used one thread. No GPU fine-tuning or GPU performance results are claimed.

## Implementation

- `ouroboros/labels.py`: shared RDKit parser and canonical labelled SMILES. Supported aliases become isotope-coded dummy atoms internally and are emitted as bracket labels, e.g. `[R1]CC`; exact and flat exact scoring preserve their identities. Supports `R`, `R1`–`R99`, primed R labels, `X`, `Ar`, common abbreviations (Ph/OMe/OEt/CO2Et/Boc/etc.), and the additional labels present in this subset (PMP/TAGa/G/M/L). Plain `*` stays an unlabelled dummy and does not equal `[R]`. Supported bare aliases and supported CXSMILES dummy labels are normalized. Unsupported/malformed annotations remain failures. `Ar` explicitly means the aryl placeholder here, rather than elemental argon; the explicit alias registry has precedence over ambiguous element names such as Ac/Pr. This is an allowlist, not a general parser for arbitrary chemical prose.
- Scoring/ingestion use that shared representation. InChI cannot represent dummy atoms: its cross-check returns unavailable/false, including for two identical labelled graphs. Fixed the previous possibility of two missing keys comparing equal. Morgan Tanimoto remains a structural similarity metric and is not a guarantee of label identity; use exact/flat exact for labelled identity.
- Tokenization accepts whole bracket aliases and normalizes supported bare aliases. `SmilesTokenizer.add_label_tokens()` / `extend()` explicitly append tokens; loading a vocabulary never modifies it. `configs/vocab.json` and released checkpoint tensors/vocabularies were not changed. Released models can load and run, but cannot emit previously unseen label tokens until fine-tuned with an expanded vocabulary.
- `ouroboros/data/rgroups.py`: seeded pendant-group substitution. Generic R/R1–R4/X collapse the smaller side of a single non-ring bond, preserving a core of at least two atoms; abbreviations collapse their exact rooted fragments. Defined stereo elements and their adjacent bonds are protected. Labels are derived from the molecule actually drawn. The renderer restores text labels after its MolBlock round-trip; internal isotope codes fit V2000's three-character fields.
- Dataset CLI adds `--rgroup-probability`, `--real-style`, `--max-tokens`. Overlength labels are counted as `label_too_long`. Generation identities include augmentation options and the renderer/augmentation/substitution/label source code, so changed recipes cannot silently reuse old shards. Clean images retain deferred, deterministically seeded degradation in the loader.
- `ouroboros/data/augment.py`: scan/background noise, small centered affine/skew, partial ink thinning/thickening, and optional binarization. These combine with existing JPEG, blur, Gaussian noise, salt/pepper, fonts, palettes, comic mode, bond widths and multiple-bond spacing. Partial thinning preserves thin text/bonds; the synthetic contact sheet was visually inspected.
- `initialize_finetune` and `train.init_checkpoint`: copy pretrained weights into a model with an append-only vocabulary, preserve every original embedding/output row, initialize new rows normally, and start a fresh optimizer/scheduler. Existing `last.pt` resume remains separate and takes precedence. Architecture/vocabulary compatibility checks fail closed.
- Leakage checks now use a namespaced canonical-label digest for dummy-containing graphs, which otherwise have no InChIKey. These digests are overlap identities only, never reported as InChI matches. Their 14-character prefixes are not InChI connectivity layers. Parent `key14` remains attached to substituted synthetic samples, preserving source-molecule split/exclusion provenance.
- `OUROBOROS_NUM_WORKERS` provides an explicit runtime DataLoader-worker override for constrained environments; ordinary configured worker counts are unchanged unless this is set.

## Re-ingestion: 20 of 22 previous failures recovered

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/workspace/venv311/bin/python scripts/ingest_real.py \
  --manifest real_data/subset100/manifest.csv --out real/subset100
```

Repeated after the final parser changes. Result: **298/300** written (99.333% label parse rate), versus 278/300 before. All **20** previously rejected R-group/abbreviation examples now parse. The remaining two malformed annotations are `[C2H2O4]...` and `[(].[)]...` (manifest rows 20 and 32). Source counts: USPTO 98, UOB 100, ACS 100. Index caches are invalidated on re-ingestion. Existing salt/charge-parent standardization remains in force, including its metal-disconnection behavior; no chemical correction of malformed source labels was attempted.

Aggregate receipts: `benchmarks/real-subset100-rgroups/{ingest_stats,label_recovery,source_hashes}.json`. Original failure details and raw predictions stay in ignored `runs/real-subset100-rgroups/`; real images/shards stay in the existing private data directories. Only aggregate reports and synthetic illustrations belong in the commit.

Evaluation manifest SHA256:
`957f0042825dab214aad2463de2974d7b4f23a1fcbf5d52ab78be2526cc3fc61`.
Downloaded parquet hashes are recorded in `source_hashes.json`; HF's default `main` revision is mutable.

## Released checkpoints: actual CPU reruns

Commands were executed separately for A, B and C (no model weights were changed):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
/workspace/venv311/bin/python scripts/predict_real.py \
  --ckpt checkpoints/ouroboros-A.pt --shards real/subset100 \
  --out runs/real-subset100-rgroups/A_preds.csv --batch-size 8

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
/workspace/venv311/bin/python scripts/eval_predictions.py \
  runs/real-subset100-rgroups/A_preds.csv \
  --json benchmarks/real-subset100-rgroups/A_report.json \
  --md benchmarks/real-subset100-rgroups/A_report.md
# Repeat with B and C.
```

Predictions were initially written in the benchmark directory, then moved to ignored `runs/` to honor DATA.md; the harness was rerun from those final paths. Every CSV's 298 gold labels was checked against the final re-ingested shard by `(source,id)`. No harness row was excluded for invalid gold.

| Arm | Correct / scored | Exact | Exact, no stereo | Valid | Mean Tanimoto | Prediction-loop seconds |
|---|---:|---:|---:|---:|---:|---:|
| A | 54 / 298 | 18.1208% | 19.1275% | 78.1879% | 0.4260 | 68.0 |
| B | 50 / 298 | 16.7785% | 16.7785% | 75.8389% | 0.4037 | 99.6 |
| C | 50 / 298 | 16.7785% | 16.7785% | 79.1946% | 0.4274 | 83.4 |

These are unchanged released models, not augmentation-trained results. Correct counts are unchanged from the earlier 278-example evaluation; the lower percentages reflect the additional 20 accepted references. Model loading/export time is outside the script's logged prediction-loop time. Times reflect this shared CPU environment, not a universal throughput claim. Full source breakdowns and bootstrap intervals are in the A/B/C JSON/Markdown reports. **Scoring parity with arXiv:2608.09100 remains unverified.**

## Augmentation throughput and synthetic contact sheet

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
/workspace/venv311/bin/python scripts/benchmark_real_aug.py \
  --n 200 --size 384 --out benchmarks/augmentation-oct29
```

Final run: 200 images, 384×384, one process, all 200 outputs labelled.

| Work | Seconds | Images/second |
|---|---:|---:|
| Post-processing only | 2.0846 | 95.9404 |
| Substitution + randomized rendering + augmentation | 2.9038 | 68.8750 |

`benchmarks/augmentation-oct29/throughput.json` contains the measured values and thread settings. `contact_sheet.png` pairs clean/augmented synthetic examples of R1, X, Ph, OMe, CO2Et and Boc. The six small example structures do not measure the entire ZINC250k pool/compose pipeline or complex-molecule rendering. The image path deliberately avoids “real” so it satisfies the repository's synthetic-figure guard.

## CPU fine-tuning smoke and tests

An actual released-A initialization smoke run built four augmented synthetic samples plus two validation samples, appended label tokens (267 total vocabulary entries), trained two batch-2 steps at 384 px, validated, and saved last/best checkpoints. Initialization: **1.0521 s**; fit including validation/checkpoint I/O: **3.3441 s**. Losses: **2.803885**, **2.432188**. This is only a functional smoke test; two validation examples and two steps do not establish accuracy or GPU throughput. Details: `benchmarks/real-subset100-rgroups/finetune_smoke.json`; private samples/checkpoints/logs: `runs/real-aug-smoke/` and `runs/real-aug-smoke.log`.

New tests cover alias canonicalization and identity, preserved real Ru/Xe atoms, malformed annotations, unavailable InChI comparisons, append-only vocabulary save/load, each abbreviation collapse/drawing round-trip, stereo protection, deterministic augmentation/morphology, deferred loader augmentation, rendering recipe mismatches, overlength-label accounting, leakage identities and pretrained embedding expansion.

Final full-suite command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OUROBOROS_NUM_WORKERS=0 PYTHONPATH=. \
/workspace/venv311/bin/python -m pytest -q
```

**356 passed, 1 skipped in 27.78 s.** Full output: `benchmarks/real-subset100-rgroups/pytest.log`.

Without the worker override, this sandbox rejects `SocketListener.setsockopt` with `PermissionError` inside PyTorch tensor sharing; multiworker tests stall. A minimal tensor DataLoader reproduced the problem. Changing worker start methods did not fix it and that experimental change was reverted. Zero workers avoids the prohibited IPC while running the entire test suite, including exact-resume checks. Normal multiworker execution was not validated in this sandbox. `ruff` was unavailable in this interpreter; no lint-pass claim is made.

Notebook validation: **nbformat.validate passed**, nbformat 4.5, 16 cells / 8 Python cells. All Python code cells and the embedded configuration-preparation program compile. Receipt: `benchmarks/real-subset100-rgroups/notebook_validation.json`.

## Exactly what still must run on Colab

Open `colab/finetune_real_aug.ipynb`, select a GPU, and execute its eight code cells in order:

1. Mount Drive; clone `https://github.com/yaniguan/ouroboros-ocsr.git`; fetch/checkout `feat/rgroups-aug-oct29`; record the source revision. These git operations are notebook instructions only and were not executed locally.
2. Install gfortran, uv, Python 3.11 and `requirements.txt`, then `--no-deps -e .`; check CUDA, run the new feature tests, fetch the chosen A/C release weights.
3. Fetch the seeded 100-per-source USPTO/UOB/ACS eval subset; ingest; record manifest/parquet hashes; save eval artifacts privately to Drive; dump eval exclusion keys. Compare hashes with the local receipt for identical-data claims.
4. Generate up to 10k ZINC250k synthetic training examples, 300 candidate val and 200 candidate test examples, seed 29, two rendering workers, `--rgroup-probability 0.5 --real-style --max-tokens 128`. Render drops are counted; checkpoint selection uses the first 200 usable val examples. Run `scripts/check_leakage.py` on real/val/test versus the rendered training shards. Save the dataset and a new GPU-runtime augmentation benchmark to Drive. No real train examples are used by default.
5. Build a vocabulary for each arm by appending aliases/observed synthetic tokens to that checkpoint's original vocabulary. Load `configs/finetune-real-aug.yaml`, replace `model` with the checkpoint's exact architecture, choose bf16 on native-bf16 hardware or fp32 on T4, and write arm-specific configs to Drive. Eval gold tokens are not used to construct the vocabulary.
6. Run the CUDA capacity probe (8 steps, batch 8, sequence 127, fp32). If memory is insufficient, lower BATCH and use a new run name before starting training.
7. Run `python -m ouroboros.train` for A and C, up to 2,000 steps each, LR 3e-5, batch 8, with synthetic-val early stopping. Select the committed best validation checkpoint; run `scripts/predict_real.py` and `scripts/eval_predictions.py` for both the release and fine-tuned model on the isolated real subset. Record actual GPU timings and accuracy; do not assume a gain.
8. ZIP reports/configs/provenance/logs/selected checkpoints, copy the ZIP to Drive, and download it. Datasets/redundant checkpoints are excluded from the ZIP. Drive root: `/content/drive/MyDrive/ouroboros-ocsr_runs/real_aug_oct29/`.

The notebook contains explicitly labelled planning estimates (T4 30–120 min, L4 20–90 min, A100 15–60 min for both arms plus setup). They are broad estimates informed by the above CPU measurements and the repository's **unmeasured** A100 assumptions in `configs/compute_assumptions.yaml`, not measured GPU results. Use the capacity probe and actual training logs to replace them. No Colab installation, source download, full 10k generation, GPU fine-tuning or post-fine-tuning real accuracy was executed here.
