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

```powershell
.\.venv\Scripts\python.exe scripts/run_p1_experiments.py --stage all
.\.venv\Scripts\python.exe scripts/plot_p1_figures.py
.\.venv\Scripts\python.exe scripts/build_p1_tex.py
tectonic -X compile papers/p1/main.tex
```

## Publication Readiness Tracker

- [x] Literature synthesis (`42` papers) & formal hypotheses (`H1.1`–`H1.3`)
- [x] Symmetry-complete $C_8 \subset SO(2)$ visual + parity-invariant $O(3)$ mirror embedding + local-Jacobian NMR symmetry orbits
- [x] `ChemVeritas-600` (`600` connectivity-disjoint scaffolds across `S1`–`S5`, strictly segregated views `V0`–`V3`, tasks `T1`–`T5`, `Veritas-RPF`)
- [x] End-to-end closed-loop neural OCSR (`Ouroboros-A/B/C/AC`, `beam=5`, `500` multi-view depictions) + `MACE-OFF23-medium` BFGS strain audit (`N=50` pairs)
- [x] Authentic published experimental $^{13}\text{C}/^1\text{H}/\text{HRMS}$ literature validation (`N=30` compounds, `15` DOI-cited isomer pairs)
- [x] 5 publication-grade vector PDF figures & compiled 2-column LaTeX manuscript (`papers/p1/main.pdf`, tag `p1-v1.0`)
