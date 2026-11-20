# Paper Branch `P1` — `p1-phys-spec-ocsr-bench`

> **PERMANENT PAPER BRANCH POLICY (`DO NOT MERGE INTO main`)**
>
> This branch (`p1-phys-spec-ocsr-bench`) is the dedicated, self-contained project branch for **Paper P1** inside `ouroboros-ocsr`.
> - **`main`** remains the clean, reusable core OCSR engine and continues evolving independently.
> - **`p1-phys-spec-ocsr-bench`** permanently preserves Paper P1's specifications, benchmarks, Bayesian physics/NMR disambiguation modules, experimental JSON/CSV receipts, vector figures, and compiled LaTeX manuscript without polluting `main` or other `pN-*` paper branches.
> - Release tag: **`p1-v1.0`** (set before the limits below were written).

## Paper Metadata

- **Paper ID**: `P1`
- **Title**: *Thermodynamically and Spectroscopically Informed Optical Chemical Structure Recognition and the `ChemVeritas-600` Multimodal Science Benchmark*
- **Target Journals**: *Chemical Science* / *JACS Au* / *Digital Discovery* / *J. Chem. Inf. Model. (JCIM)*
- **Compiled Manuscript**: `papers/p1/main.pdf` (`papers/p1/main.tex`, `papers/p1/references.bib` — 56 BibTeX entries)
- **Spec & Literature Review**: `specs/001-phys-spec-ocsr-bench/` (`42` verified 2019–2026 papers in `research.md`)
- **Packages & Scripts**:
  - `ouroboros/phys/` (`hrms.py`, `epimers.py`, `strain.py`, `nmr.py`, `posterior.py`, `lit_nmr_panel.py`)
  - `ouroboros/si/` (`parser.py`)
  - `ouroboros/bench/` (`catalog.py`, `builder.py`, `scoring.py`, `vlm_harness.py`)
  - `scripts/run_p1_experiments.py`, `scripts/plot_p1_figures.py`, `scripts/build_p1_tex.py`
- **Experimental Receipts & Vector Figures**:
  - `benchmarks/p1_chemveritas/neural_ocsr_closed_loop.json` & `neural_ocsr_by_view.csv`
  - `benchmarks/p1_chemveritas/mace_strain_audit.json`
  - `benchmarks/p1_chemveritas/gold600_ablation_receipt.json` & `gold600_summary_table.csv`
  - `benchmarks/p1_chemveritas/tier2_exp_nmr_validation.json`
  - `benchmarks/p1_chemveritas/figures/fig1_closed_loop_neural_ocsr.pdf` .. `fig5_tier2_exp_nmr.pdf`

## Reproducibility Commands

```bash
python scripts/run_p1_experiments.py --stage all
python scripts/plot_p1_figures.py
python scripts/build_p1_tex.py
tectonic -X compile papers/p1/main.tex
```

## What the numbers mean

Read these limits before quoting any P1 number.

| Receipt | Images | Spectra | Note |
|---|---|---|---|
| `gold600_ablation_receipt.json` (ChemVeritas-600, A0–A6) | None. Stage 4 builds the benchmark with `render_images=False`. | Noise-free: 13C/1H peaks are the exact `predict_nmr` output, rounded to 0.1/0.01 ppm (`ouroboros/si/parser.py`, `format_si_paragraph`). The only jitter is a ±1.6 ppm HRMS mass error. | The T2 "visual" score is a fixed rank prior `-0.02 * i` over a shuffled pool of 6 candidates (`ouroboros/bench/vlm_harness.py`), so A0 (19.7%) is close to chance (1/6). Builder and scorer use the same `predict_nmr` and strain settings, so the true candidate matches the SI spectrum up to rounding, and the A4–A6 gain (to 77.0%) is an in-model upper bound. T4 reaches 100% at A3+ by construction: the reference energies and the A3+ predictions come from the same MMFF94s `evaluate_3d_strain` call. T5 abstention is 100% by construction once the parity guard is on (A1+). |
| `neural_ocsr_closed_loop.json` (100 scaffolds, 5 views) | Synthetic RDKit depictions in 5 views (clean, exact 90° rotation, off-grid rotation, crowded-scan degradation, condensed layout), not images from papers. Pretrained Ouroboros-A/B/C checkpoints (not in the repo); AC is a rescored A+C beam ensemble. | Same as Gold-600: exact `predict_nmr` output; only HRMS is jittered (±1.5 ppm). | Tests the real vision models; the spectra part is still in-model. Stage 3 cannot be rerun from a fresh checkout without the checkpoints. |
| `tier2_exp_nmr_validation.json` (30 compounds) | None. | 13C shifts, DEPT counts and one 1H multiplet per compound transcribed from the cited papers. HRMS m/z is computed from the truth formula + 0.8 ppm, and the truth formula is given to the scorer. | The only test on measured spectra. Every beam places the truth last (prior −0.55/−0.58, behind both decoys), so A0–A2 are 0/30 by design and A3+ must overcome a fixed prior penalty. |
| `mace_strain_audit.json` (50 pairs) | None. | None. | 30 of 50 corrupt structures fail RDKit embedding and get a fixed 45 kcal/mol penalty, not a MACE energy. Geometry: MMFF94s pre-optimization (80 iterations), then at most 8 MACE BFGS steps. The split summary is in `mace_strain_audit_summary.json` below. |

The manuscript text (`papers/p1/main.tex`) has been updated to state these limits.

## Strain audit summary

`python scripts/summarize_strain_audit.py` writes `benchmarks/p1_chemveritas/mace_strain_audit_summary.json` from the committed receipt.

- 30 of 50 corrupt structures fail RDKit embedding and carry a fixed 45 kcal/mol penalty, so the 45.00 kcal/mol median in the earlier manuscript draft was the penalty constant.
- The 92.0% preference rate and the ROC-AUC values (0.853 MACE, 0.919 MMFF) also include these 30 penalty pairs.
- On the 20 pairs where both structures embed (MMFF94s pre-optimization, then at most 8 MACE BFGS steps), the MACE-OFF23 difference has median 1.97 kcal/mol (min -1.14, max 4.93). The corrupt structure is higher in 16 of 20 pairs (80%).
- Per stratum: S1, S3 and S5 have no measured pair (all 10 fail embedding). S2 and S4 have 10 measured pairs each, with median 2.25 and 1.57 kcal/mol.

The manuscript strain text and Table 2 now use this summary.

## Publication Readiness Tracker

Status: draft. The items below are done as code and receipts; see "What the numbers mean" for their limits. Tag `p1-v1.0` predates this section; a new tag is needed after these fixes.

- [x] Literature synthesis (`42` papers) & formal hypotheses (`H1.1`–`H1.3`)
- [x] Symmetry-complete $C_8 \subset SO(2)$ visual + parity-invariant $O(3)$ mirror embedding + local-Jacobian NMR symmetry orbits
- [x] `ChemVeritas-600` (`600` connectivity-disjoint scaffolds across `S1`–`S5`, strictly segregated views `V0`–`V3`, tasks `T1`–`T5`, `Veritas-RPF`)
- [x] End-to-end closed-loop neural OCSR (`Ouroboros-A/B/C/AC`, `beam=5`, `500` multi-view depictions) + `MACE-OFF23-medium` BFGS strain audit (`N=50` pairs)
- [x] Published $^{13}\text{C}/^1\text{H}$ literature validation (HRMS simulated) (`N=30` compounds, `15` DOI-cited isomer pairs)
- [x] 5 publication-grade vector PDF figures & compiled 2-column LaTeX manuscript (`papers/p1/main.pdf`, tag `p1-v1.0`)
