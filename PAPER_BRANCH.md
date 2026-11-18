# Paper Branch `P1` — `p1-phys-spec-ocsr-bench`

> **PERMANENT PAPER BRANCH POLICY (`DO NOT MERGE INTO main`)**
>
> This branch (`p1-phys-spec-ocsr-bench`) is the dedicated, self-contained project branch for **Paper P1** inside `ouroboros-ocsr`.
> - **`main`** remains the clean, reusable core OCSR engine and continues evolving independently.
> - **`p1-phys-spec-ocsr-bench`** permanently preserves Paper P1's specifications, benchmarks, Bayesian physics/NMR disambiguation modules, experimental JSON/CSV receipts, vector figures, and compiled LaTeX manuscript without polluting `main` or other `pN-*` paper branches.
> - Immutable release tag: **`p1-v1.0`**.

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
| `gold600_ablation_receipt.json` (ChemVeritas-600, A0–A6) | None. Stage 4 builds the benchmark with `render_images=False`. | Simulated with `predict_nmr` plus small jitter (`ouroboros/bench/builder.py`). | The T2 "visual" score is a fixed rank prior `-0.02 * i` over a shuffled pool of 6 candidates (`ouroboros/bench/vlm_harness.py`), so A0 (19.7%) is close to chance (1/6). The posterior scores candidates with the same `predict_nmr`, so the A4–A6 gain (to 77.0%) is an in-model upper bound. T5 abstention is 100% by construction once the parity guard is on (A1+). |
| `neural_ocsr_closed_loop.json` (100 scaffolds, 5 views) | Real rendered images, real pretrained Ouroboros-A/B/C/AC weights. | Simulated with `predict_nmr`, same as above. | Tests the real vision models; the spectra part is still in-model. |
| `tier2_exp_nmr_validation.json` (30 compounds) | None. | Published experimental values. | The only test on measured spectra. The candidate prior puts a decoy first, so A0–A2 are 0/30 by design. |
| `mace_strain_audit.json` (50 pairs) | None. | None. | 30 of 50 corrupt structures fail RDKit embedding and get a fixed 45 kcal/mol penalty, not a MACE energy. Relaxation is BFGS with 8 steps. See issue #92 for the split summary. |

The manuscript text needs to be updated to state these limits before submission.

## Publication Readiness Tracker

Status: draft. The items below are done as code and receipts; see "What the numbers mean" for their limits.


- [x] Literature synthesis (`42` papers) & formal hypotheses (`H1.1`–`H1.3`)
- [x] Symmetry-complete $C_8 \subset SO(2)$ visual + parity-invariant $O(3)$ mirror embedding + local-Jacobian NMR symmetry orbits
- [x] `ChemVeritas-600` (`600` connectivity-disjoint scaffolds across `S1`–`S5`, strictly segregated views `V0`–`V3`, tasks `T1`–`T5`, `Veritas-RPF`)
- [x] End-to-end closed-loop neural OCSR (`Ouroboros-A/B/C/AC`, `beam=5`, `500` multi-view depictions) + `MACE-OFF23-medium` BFGS strain audit (`N=50` pairs)
- [x] Authentic published experimental $^{13}\text{C}/^1\text{H}/\text{HRMS}$ literature validation (`N=30` compounds, `15` DOI-cited isomer pairs)
- [x] 5 publication-grade vector PDF figures & compiled 2-column LaTeX manuscript (`papers/p1/main.pdf`, tag `p1-v1.0`)
