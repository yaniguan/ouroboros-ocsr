# TASK.md — Ouroboros live status

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (with evidence) · `[!]` blocked/failed.
Local dev machine: 4 CPU cores, 15 GB RAM, no GPU, Python 3.11.15.
Amendment 1 (2026-09-25, real data / synthetic-to-real framing) is merged below; items it added
or changed are tagged **(Am1-A … Am1-F)**.

## Needs user

### U0 — Phase 0 Colab smoke run  *(blocks: Phase 0 [colab] item only; nothing else waits on it)*
1. Open `notebooks/00_colab_setup.ipynb` in Colab (File → Open notebook → GitHub →
   `yaniguan/ouroboros-ocsr`, branch `claude/vigilant-johnson-j4882f`).
2. Runtime → Change runtime type → **A100 GPU**.
3. If the repo is private: add Colab secret `GITHUB_TOKEN` (read access to this repo) and enable it
   for the notebook.
4. Runtime → Run all. Expected wall time ≈ 5–10 min (most of it is building `py3nj` and pip installs).
5. **Report back:** (a) whether every cell ran without error, (b) the final `wall time: X min`
   line, (c) the output of the last cell (`pip freeze` subset) and the `nvidia-smi` line,
   (d) the pytest summary line.

### U1 — Real training / evaluation data from arXiv:2608.09100 (Am1-A)
- Where are the real training sets and the real-document eval sets (ACS, CLEF-IP, USPTO per the
  paper's abstract) stored, in what format (image paths + SMILES? CSV/JSONL columns?), and **may
  they be used in this project** (license / redistribution)?
- Until then the adapter is built and tested against a synthetic stand-in manifest.

### U2 — Real-data fractions of the paper's grid (Am1-B)
- The abstract mentions 0, 9.5% and 50.2% real data for Qwen2.5-VL. Please give the full list of
  fractions used, so the Ouroboros grid is directly comparable. Placeholders in use:
  `{0, 0.05, 0.1, 0.25, 0.5, 1.0}` (marked as placeholders in `configs/`).

### U3 — Paper's scoring / canonicalization code (Am1-C)
- Path or repo of the scoring code used in arXiv:2608.09100 / VERDICT, for per-sample parity checks.
- Unverified details I could not read (arxiv.org is blocked from this sandbox; only search-engine
  snippets were visible): which RDKit standardization calls are used for "neutral largest fragment"
  (e.g. `rdMolStandardize.ChargeParent` vs. `LargestFragmentChooser`+`Uncharger`), whether
  tautomers are canonicalized, whether explicit/implicit H or isotopes are normalized, how
  unparsable predictions count in exact match (as wrong, presumably), and how InChIKey identity is
  combined with exact match in the tables.

### U4 — Generate the 1M dataset on Colab (Phase 1 [colab]; 200k = its first 200 shards)
1. Open `notebooks/01_generate_data.ipynb` (branch `claude/vigilant-johnson-j4882f`). It needs no GPU;
   any runtime with many vCPUs works (an A100 runtime has ~12).
2. Run all. Expected: pool ≈ 5 min, composition ≈ 15 min (single-threaded), rendering 1M ≈ 20–30 min
   on 12 vCPUs, copy to Drive ≈ 5–15 min. Output: `MyDrive/ouroboros/data/full/` ≈ 7 GB
   (1M train ≈ 6.8 GB, 200k subset ≈ 1.4 GB, val+test ≈ 0.1 GB). Needs ~8 GB free on Drive.
3. **Report back** the last cell's output: total and 200k-subset sizes, `pool.stats.json`,
   `compose.json`, the three `render` lines (samples, drops, seconds) and the wall times.

### U5 — Decisions needed on the experiment grid (Am1-E) — see "Compute estimate" in Phase 4
- The full grid (4 arms × 6 placeholder real fractions × {50k, 200k} × 3 seeds = 132 runs) is
  estimated at **150.9 A100-h** (assumed, not yet measured throughput), just over the 150 h limit.
- **Proposed pruned grid (72 runs, ≈ 85 A100-h):** real fractions {0, 0.1, 0.5}, all 4 arms, both
  sizes, 3 seeds. Why it still tests H1–H3: f = 0 is the synthetic-only condition for H1 (rendered
  and rotated sets) and H2 (real sets); 0.1 and 0.5 bracket low and high real supervision for H3 and
  sit next to the 9.5% / 50.2% points in arXiv:2608.09100's abstract (to be replaced by your exact
  fractions, U2). Seeds stay at 3 so the CIs remain meaningful.
- Steerable arm config: param-matched vs FLOP-matched (numbers in Phase 3). My recommendation:
  FLOP-matched in the main grid (equal compute per image, the practical constraint), param-matched
  as a single ablation (one size, f = 0, 3 seeds), because the param-matched C8 costs 12.4× the
  FLOPs (127 vs 10.3 GFLOPs per image).
- Please reply: full or pruned grid, and FLOP- vs param-matched.

## Phase 0 — Scaffold

- [x] `pip install -e .` succeeds in a fresh virtualenv; exact versions pinned in requirements.
  — fresh venv: `pip install -r requirements.txt` (81 exact pins, 2m07s from pip cache) +
  `pip install -e .` OK; pytest 7 passed and ruff clean inside it, 2026-09-25.
- [x] `pytest` passes, ≥ 1 smoke test per subpackage (≥ 6 tests). — 7 passed (6 subpackages +
  third-party imports), 2026-09-25.
- [x] `ruff check .` reports 0 errors. — "All checks passed!", 2026-09-25.
- [x] README: research question, symmetry-per-stage table, no "first"/"groundbreaking" claims,
  related-work TODO lists DECIMER, MolScribe, MolNexTR, MolSight. — `grep -iE 'first|groundbreak'
  README.md` → no matches, 2026-09-25. *(Modified by Am1-F: research question replaced, exactly one
  sanctioned "first" sentence allowed; tracked under Am1-F below.)*
- [x] Colab notebook exists (`notebooks/00_colab_setup.ipynb`, generated by
  `scripts/make_colab_setup_nb.py`): mounts Drive, clones repo, installs deps, copies shards to
  `/content`, runs `pytest -q`. — 2026-09-25.
- [ ] [colab] Notebook runs top to bottom on A100; `import escnn, mace` succeeds; wall time < 15 min.
  → see Needs user U0.

## Phase 1 — Data pipeline

Sources: ZINC250k (GitHub raw) + MOSES (GitHub LFS media), 2,186,417 input SMILES → pool of
2,154,733 unique constitutions (785 charged after neutralization, 30,899 duplicates), 612 s on 4
cores (`data/full/pool.stats.json`).

- [x] Molecule filter: H C N O F P S Cl Br I only, 5–60 heavy atoms, neutral; unit-tested. —
  `ouroboros/data/filter.py`; `tests/test_data.py::test_filter_rules` (11 cases incl. both bounds,
  Si/B/Se, radical, quaternary N, isotope) + salt/neutralization test, all pass, 2026-09-25.
- [x] Stereo composition ≥ 30% (configurable), measured within ±2 pp of target. — target 0.40
  (`--stereo-fraction`); measured 0.400 for train prefixes 10k/50k/200k/1M, val 0.400, test 0.400
  (recomputed from label SMILES for the 50k prefix: 0.400) → `benchmarks/data/stereo_fraction.json`, 2026-09-25.
- [x] Splits: 0 InChIKey overlap train/val/test (script-verified). — `scripts/verify_data.py splits`:
  1M/5k/10k, full-key and connectivity-block overlaps all 0 → `benchmarks/data/splits.json`, 2026-09-25.
- [x] Label integrity: 100% parse; canonical label == canonical source on 10k sample. — 12,000/12,000
  (10k train + 1k val + 1k test; 0 parse failures, 0 mismatches) → `benchmarks/data/labels.json`;
  additionally 0 render-time drops (label ≠ source) among all 65,000 rendered samples, 2026-09-25.
- [x] Mirror test: 200/200 mirrored renderings labeled as the enantiomer. — 200/200 (chiral, non-meso
  test molecules, random styles) → `benchmarks/data/mirror.json`, 2026-09-25.
- [x] Style randomization (fonts, line width, bond length, label style, noise, blur, JPEG);
  contact sheet `benchmarks/data/contact_sheet.png`. — 64 samples cover 10 fonts, 4 palettes, line
  width 1.1–4.0 px, bond length 18–39 px, 16 blurred / 23 noisy / 16 JPEG, 7 explicit-methyl, 3 comic
  → `benchmarks/data/contact_sheet.json`, 2026-09-25.
- [x] Rendering throughput ≥ 20 img/s/process at 384 px. — 55.8 img/s single process (style +
  render + degradation + PNG) → `benchmarks/data/render_speed.json`; shard generation 346 img/s with
  4 processes, 2026-09-25.
- [x] Loader throughput ≥ 500 img/s with 8 workers. — 605 img/s (8 workers on this 4-core VM, incl.
  load-time degradation, shuffled random access, batch 64) → `benchmarks/data/loader_speed.json`, 2026-09-25.
- [x] 10k and 50k shards generated locally; 200k / 1M scripts tested on 1k dry run. — 50k train
  (50 shards; 10k = first 10 shards) + 5k val + 10k test in `data/full/shards`, 443 MB (train 341 MB,
  ≈6.8 KB/img), 0 drops, train 144.6 s on 4 processes. Dry runs `--preset 200k/1m --dry-run 1000`:
  compose + render 1000/100/100 OK, 0 drops (207 s / 877 s incl. composing the full manifest); the
  dry-run 1M manifest is byte-identical to `data/full` and its first 200k rows equal the 200k
  manifest (nested prefixes, deterministic), 2026-09-25.
- [ ] [colab] 200k and 1M shards on Drive; sizes and times reported. → Needs user U4. *(Am1-E grid uses only 50k and
  200k; 1M is kept for the original data-efficiency curve unless you drop it.)*

### Am1-A — Real-data adapter
- [x] Adapter ingests a manifest (image path, SMILES, source/split fields) into the synthetic shard
  format; interface documented. — `ouroboros/data/real.py` (docstring + DATA.md),
  `scripts/ingest_real.py`; `tests/test_real.py` (5 pass); stand-in set (600 cropped synthetic val
  images, PNG/JPEG, varying sizes) ingested 600/600, 2026-09-25.
- [~] 100% of real labels parse after Am1-C standardization; failures logged with reasons, never
  silently dropped. — mechanism done and tested (every failed row → `ingest_failures.jsonl` with
  reason, counted in `ingest_stats.json`; test covers unparsable/empty label and missing image);
  the 100% measurement needs the real sets (U1).
- [~] Leakage check: 0 InChIKey overlap between every real-document eval set and every training
  set (synthetic + real); script prints counts and exits nonzero if any > 0. — `scripts/check_leakage.py`
  (exit 1 on overlap, tested with a planted duplicate); `--dump-eval-keys` + `build_dataset.py
  --exclude-keys` / `ShardDataset(exclude_key14=)` remove eval molecules (incl. stereoisomers) from
  training. Stand-in run: 0 overlaps for all 4 pairs. Real-set measurement pending U1.
- [x] Real data / images / derived shards never committed: `.gitignore` + pre-commit check;
  `DATA.md` notes redistribution restrictions. — `scripts/check_no_data.py` installed as
  `.git/hooks/pre-commit` (`scripts/install_git_hooks.sh`, also `.pre-commit-config.yaml`); a staged
  `benchmarks/real_tmp/x.png` was refused ("Commit refused"), rules unit-tested, 2026-09-25.
- [!] Real sets from the paper — waiting on U1.

## Phase 2 — Baseline model and training infrastructure
- [x] Baseline encoder + 6-layer Transformer decoder; 20M–60M params. — 43.17M total (encoder
  17.81M: ResNet-18-style CNN + abs. 2D pos-emb + 2-layer mixer; decoder 25.36M: 6 layers, d=512,
  8 heads, ff 2048, 133-token vocab, tied embeddings); encoder 10.3 GFLOPs @384 px, 2026-09-25.
- [x] Overfit: 256 samples → ≥ 99% exact match within 3,000 steps. — 256/256 = 100% at step 1,750
  (13.3% @250, 55.9% @500, 83.2% @750, 91.0% @1250); settings: default architecture (43.1M) at
  128 px, CPU, fp32, batch 32, AdamW lr 5e-4, warmup 100, cosine over 3,000, dropout 0, greedy
  decoding, eval every 250 steps; 57 min → `benchmarks/train/overfit_baseline.json`, 2026-09-25.
- [x] Resume test: ≤ 1% mean relative loss difference over next 100 steps. — default architecture
  (43.1M, dropout 0.1, rotation aug on, 2 loader workers) at 128 px, CPU: checkpoint at step 50,
  run killed at 73, resumed at 50; losses of steps 51–150 identical to the uninterrupted run (mean
  and max relative difference 0.0) → `benchmarks/train/resume_baseline.json`; restored: model,
  optimizer, LR scheduler, python/numpy/torch RNG, sampler position, augmentation RNG; also
  bit-exact in `tests/test_train.py`, 2026-09-25.
- [x] Checkpoint save < 30 s. — default model @384 px with Adam state: 518 MB, save 0.56–1.78 s
  (3 saves), load 0.42 s, local disk → `benchmarks/train/checkpoint_time.json`, 2026-09-25.
- [x] Metric unit tests 100% pass (stereo-aware exact match, InChI match, Tanimoto,
  per-stereocenter accuracy, invalid-SMILES rate). *(Am1-C: standardization below applies.)* —
  `tests/test_metrics.py` 16/16 pass (salts, enantiomer, missing stereo, 1-of-2 centres, E/Z, ring
  cis/trans, random SMILES orderings ×6 molecules, invalid/empty, tautomer InChIKey, aggregation,
  bootstrap), 2026-09-25.
- [x] Rotation-sweep eval (0–360°, 15°), on-grid vs off-grid, tested on dummy model.
  *(Am1-D: must run on rendered AND real-document eval sets.)* — `ouroboros/eval/evaluate.py`,
  `scripts/evaluate.py`; dummy exactly-C4-invariant model: exact 1.0 at the 4 pixel-exact angles, 0.0
  at the 20 others, flags for C4/C8/C16 grids; `tests/test_rotation_sweep.py` 2/2, 2026-09-25.
- [ ] [colab] Baseline on 200k: exact match ≥ 70%, invalid < 5%.

### Am1-B — Mixture sampler
- [x] Configurable real fraction per batch; measured fraction over 10,000 samples within ±1 pp;
  identical sample order across two runs with the same seed. — `ouroboros/data/mixture.py`; over
  10,000 samples (batch 64): f=0→0.0000, 0.05→0.0499, 0.1→0.1000, 0.25→0.2502, 0.5→0.5000,
  1.0→1.0000 (max |Δ| 0.02 pp); two runs identical for every f; every batch within one sample of
  B·f; resume at arbitrary positions exact (`tests/test_mixture.py` 20/20), 2026-09-25.
- [x] Fraction 0 reproduces the synthetic-only sample order exactly (same seed). — 3,000-sample
  stream identical to `ResumableSampler` (test), 2026-09-25.
- [!] Paper's fraction grid — waiting on U2 (placeholders `{0, 0.05, 0.1, 0.25, 0.5, 1.0}`).

### Am1-C — Scoring parity with arXiv:2608.09100
- [x] Standardization: RDKit parse + canonicalize; salts/solvates → neutral largest fragment;
  exact match under full stereochemistry; identity cross-check via stereo-preserving InChIKey;
  validity = fraction parseable by RDKit. Verification status per rule recorded in
  `ouroboros/eval/standardize.py` docstring (see Decisions log). — implemented with
  `rdMolStandardize.ChargeParent`; the RDKit call itself and two choices (empty output = invalid, no
  tautomer canonicalization) are marked UNVERIFIED vs. the paper, 2026-09-25.
- [!] pending parity — per-sample agreement with the paper's scoring code on ≥ 1,000 pairs (100%
  identical) needs U3. Every results table carries the footnote "scoring parity with
  arXiv:2608.09100 unverified" until this is done.

### Am1-D — Evaluation reporting
- [x] Rendered and real-document test sets reported as separate columns; aggregation raises an
  error on any headline number averaged across rendered and real sets. — `ouroboros/eval/aggregate.py`
  (`MixedKindsError` from `pool_sets` / `--average`), tested on synthetic logs, 2026-09-25.
- [x] Rotation sweep runs on both rendered and real-document eval sets. — every `eval.sets` entry
  (kind rendered|real) gets angle 0 in full + the 24-angle sweep on a fixed prefix
  (`sweep_max_samples`, default 2000); tested with a rendered and a (pretend) real set, 2026-09-25.
- [x] Per-set sample counts and bootstrap 95% CIs (≥ 1,000 resamples) for exact match; plots draw CIs.
  — table cells show mean ± std over seeds, [95% CI from 1,000 paired item resamples], n; `results_table`
  refuses n_boot < 1000; all plots draw CI bands, 2026-09-25.

## Phase 3 — Steerable CNN encoder
- [x] escnn C_N encoder (N ∈ {4, 8, 16}), regular reps, group pooling, invariant relative-position tokens.
  — `ouroboros/encoder/steerable.py` (`rot2dOnR2`, no flips; stride-1 R2Conv + blur/2×2 pooling;
  GroupPooling; distance-biased token mixer; `TokenHead` slot for arm D), 2026-09-25.
- [x] C4 equivariance < 1e-4; invariant token set equal up to permutation within 1e-4. — 384 px,
  full-width random-init encoders, eval mode, fp32 (`scripts/measure_equivariance.py` →
  `benchmarks/encoders/equivariance.json`): C4 feature maps max rel. err 8.5e-07,
  token set (after the known grid permutation) 8.2e-07 over 90/180/270°; the same
  holds for C8 (1.2e-06 / 5.8e-07) and C16 (1.4e-06 /
  7.4e-07) at 90° multiples. Also unit-tested (`tests/test_equivariance.py`, 15 pass,
  incl. end-to-end decoder logits invariant for C4, and NOT reflection invariant), 2026-09-25.
- [x] C8/C16 off-grid equivariance error recorded. — escnn's interpolated action on input and
  output, central disk: C8 at 45/135/225/315°: feature rel. err 0.165–0.165, mean-token
  rel. change 0.0037–0.0037; C16 at the 12 non-90° multiples of 22.5°: feature 0.192–0.240,
  mean token 0.0006–0.0008 (report only; includes pixel-interpolation error of both sides),
  2026-09-25.
- [x] Param-matched (±10%) and FLOP-matched (±10%) configs; throughput recorded. —
  `scripts/profile_encoders.py` → `benchmarks/encoders/profile.json` (encoder only, 384 px, FLOPs
  by `FlopCounterMode`, CPU fp32 throughput on this VM; GPU throughput comes from Colab):

  | encoder | fields / widths | params | vs base | GFLOPs | vs base | CPU img/s (eval) |
  |---|---|---|---|---|---|---|
  | baseline (A/B) | 64-128-256-512 ch | 17.81M | — | 10.27 | — | 15.9 |
  | C8 param-matched | 28-56-111-222 regular fields | 17.71M | −0.6% | 126.98 | +1136% | 1.6 |
  | C8 FLOP-matched | 7-14-27-55 regular fields | 7.02M | −60.6% | 10.29 | +0.2% | 10.8 (8.0 train-mode) |

  Both include the identical 2-layer token mixer (≈6.3M). The sweep uses the FLOP-matched config
  pending U5, 2026-09-25.
- [ ] Overfit test passes.
- [x] `.export()` matches training model within 1e-4; speedup recorded. — FLOP-matched C8 @384 px
  on 4 rendered test images, eval mode: max rel. err 0.0e+00; CPU throughput
  5.23 (escnn train-mode) / 5.73 (escnn eval) /
  6.24 img/s (exported) → speedup 1.19× / 1.09×
  (`benchmarks/encoders/export.json`); evaluation uses the exported encoder, 2026-09-25.
- [ ] [colab] C8 on 200k vs baseline.

## Phase 4 — Experiment runner
- [!] ~~Sweep configs A/B/C/C+ × {10k, 50k, 200k, 1M} × ≥ 3 seeds from one definition.~~
  Superseded by Am1-E (grid redefined), 2026-09-25.
- [x] Aggregation script (table + plots) tested on synthetic logs. *(Am1-D rules apply.)* —
  `scripts/aggregate.py`: results.md/json, data-efficiency, real-fraction and rotation-sweep plots;
  `tests/test_aggregate.py` 2/2 on 216 fake runs' logs, 2026-09-25.
- [ ] [colab] Sweep executed.

### Am1-E — Experiment grid (replaces the Phase 4 sweep definition)
- [x] Grid = arms {A, B, C, C+} (D/E when Phase 5 lands) × real fraction (Am1-B) × size
  {50k, 200k} × ≥ 3 seeds, generated from one sweep file. — `configs/sweep.yaml` →
  `scripts/make_sweep.py` → 132 configs in `configs/sweep/` + `index.csv` (4 arms × 6 placeholder
  fractions × 2 sizes × 3 seeds, f = 1.0 kept at one size only since it uses no synthetic data);
  generator refuses arm overrides of the decoder and asserts one decoder config, 2026-09-25.
- [ ] Every config passes a 50-step dry run.
- [x] Compute estimate (GPU-h per run and total) in TASK.md; if > 150 A100-h, propose a pruned grid
  that still tests H1–H3 under Needs user. — see "Compute estimate" below; full grid 150.9 A100-h
  (> 150) → pruned grid proposed in U5, 2026-09-25.

#### Compute estimate (2026-09-25; `scripts/estimate_compute.py`)
Assumptions (NOT measured; `configs/compute_assumptions.yaml`): A100 bf16, 384 px, batch 64;
training 600 img/s for the baseline, 400 img/s for the steerable C8 (FLOP-matched), 300 img/s for
arm D; eval 58k greedy decodes at 1000 img/s; 0.05 h overhead per run. Steps: 20k (50k set),
40k (200k set) → 1.28M / 2.56M samples seen.

| arm | size | runs | GPU-h/run | GPU-h |
|-----|------|------|-----------|-------|
| A, B (each) | 50k | 18 | 0.66 | 11.9 |
| A, B (each) | 200k | 15 | 1.25 | 18.8 |
| C, C+ (each) | 50k | 18 | 0.95 | 17.2 |
| C, C+ (each) | 200k | 15 | 1.84 | 27.7 |
| **full grid** | | **132** | | **150.9** |
| pruned grid (fractions {0, 0.1, 0.5}) | | 72 | | 84.8 |

Arm D (Phase 5) would add ≈ 2.3 h per 200k run. The estimate will be redone with the img/s that the
first Colab runs log (`img_per_s` in `log.jsonl`).

## Phase 5 — Equivariant attention (D) and canonicalization (E, optional)
- [ ] Steerable stem + group-equivariant self-attention with rotated relative positions.
- [ ] Equivariance thresholds as Phase 3; overfit passes.
- [ ] Memory estimate: batch ≥ 32 @ 384 px bf16 fits 40 GB.
- [ ] Arm E invariance < 1e-4 (if implemented).
- [ ] [colab] D (and E) in sweep.

## Phase 6 — Geometry / MACE (positioned as an error-propagation analysis, Am1-F)
- [ ] SMILES → N ETKDG conformers → MACE-OFF opt → lowest-energy conformer (energy, forces).
- [ ] Embedding success ≥ 98% on 1,000 molecules; failures logged.
- [ ] Stereo preservation ≥ 99% on 1,000 chiral molecules.
- [ ] Convergence (fmax < 0.05 eV/Å) ≥ 95%; median time recorded.
- [ ] Enantiomer pairs |ΔE| < 1e-3 eV (20 pairs).
- [ ] Error categorization unit tests 100%.
- [ ] Error-propagation script tested on a small set.
- [ ] [colab] Energy-error distribution per category on best arm's predictions.

### Am1-F — README / related work
- [x] Research question replaced; H1–H3 pre-registered with date (before any Colab result). —
  README "Pre-registered hypotheses (registered 2026-09-25 …)"; no Colab run has happened yet.
- [x] Related work cites arXiv:2608.09100 and VERDICT (motivating), DECIMER, MolScribe, MolNexTR,
  MolGrapher, MolSight, MolParser, DeepMoLM, Auto3D. — README "Related work", 2026-09-25
  (VERDICT arXiv id 2608.22183 found via web search; paper bodies not readable from sandbox).
- [x] Only novelty statement: "To our knowledge, the first systematic study of group-equivariant
  vision encoders for OCSR, evaluated against both augmentation and real-data supervision." —
  `grep -niE 'first|groundbreak' README.md` → exactly line 23 (that sentence), 2026-09-25.
- [x] Geometry/MACE stage positioned as error-propagation analysis, not a novel pipeline. —
  README "Geometry / MACE stage", 2026-09-25.

## Decisions log

- 2026-09-25 — torch pinned to 2.10.0 locally (PyPI CUDA wheel; the pytorch.org CPU index is
  blocked by the sandbox proxy). On Colab the preinstalled CUDA torch is kept
  (`requirements-colab.txt` = `requirements.txt` minus torch/triton/nvidia/cuda wheels) to avoid a
  multi-GB download and driver mismatch; the notebook prints the torch version used.
- 2026-09-25 — numpy is pinned to 1.26.4: `lie_learn` (escnn dependency) and `matscipy`
  (mace-torch dependency) require numpy < 2.
- 2026-09-25 — Colab notebook runs project code in subprocesses (`!python ...`) so the pinned
  numpy is picked up without a kernel restart.
- 2026-09-25 — Notebooks are generated from `scripts/make_*_nb.py` so they diff cleanly in review.
- 2026-09-25 — Molecule sources: ZINC250k + MOSES (both reachable from the sandbox; ChEMBL/PubChem
  FTP are blocked). Both are ZINC-derived and drug-like; the largest molecules have < 60 heavy
  atoms, so the upper bound of the filter is rarely active. Other sources can be added in
  `ouroboros/data/sources.py`.
- 2026-09-25 — "Neutral" = net formal charge 0 and no radicals, evaluated after standardization
  (largest organic fragment + RDKit `Uncharger`), so protonated amines / carboxylate salts are
  neutralized instead of discarded; charge-separated neutral groups (nitro) are kept.
- 2026-09-25 — Labels are re-derived from the drawing (2D coords + wedge flags → MolBlock → RDKit
  stereo perception), and the drawn molecule is that same MolBlock molecule. Images and labels
  therefore cannot disagree; the mirror test exercises exactly this path.
- 2026-09-25 — Every drawing depicts a definite geometry for stereogenic double bonds, so E/Z is
  always assigned; tetrahedral stereo is assigned (randomly, seeded by the InChIKey) only for
  molecules drawn into the stereo bucket. Train order is interleaved so every prefix has the target
  stereo fraction (default 0.40) → 10k ⊂ 50k ⊂ 200k ⊂ 1M are nested prefixes of one manifest.
- 2026-09-25 — Splits are assigned by a hash of the InChIKey connectivity block, so all
  stereoisomers of a constitution share a split (stricter than full-key disjointness).
- 2026-09-25 — The molecule is drawn inside the inscribed circle of the canvas so any rotation
  about the image centre never crops it; image tensors use ink = 1 − gray/255 (background 0) so
  zero-fill rotation adds no border.
- 2026-09-25 — Training reads the WebDataset tar shards by random access (tar member offsets
  indexed once) instead of streaming, so sample order is a pure function of (seed, position) and
  resume is exact. Shards remain valid WebDataset shards.
- 2026-09-25 — Am1-C: from search-engine snippets of arXiv:2608.09100 (full text blocked here), the
  following are confirmed: RDKit parse + canonicalize; exact match requires equality under the
  complete stereochemistry convention; identity additionally checked via InChIKey; validity =
  fraction parseable by RDKit; salts/solvates reduced to the neutral largest fragment, mapped to a
  stereo-preserving InChIKey; real benchmarks ACS, CLEF-IP, USPTO. Everything else is unverified (U3).

## Deviations

- Added `scripts/` (notebook generators and data-generation CLIs) to the proposed repo layout.
- Am1 (2026-09-25): research question reframed around the synthetic-to-real gap; Phase 4 sweep
  redefined (sizes {50k, 200k} × real fractions instead of {10k, 50k, 200k, 1M}); README now allows
  exactly one sanctioned "first" sentence (overrides the Phase 0 "no first claims" wording);
  real-data adapter, mixture sampler, scoring parity and split reporting added.
