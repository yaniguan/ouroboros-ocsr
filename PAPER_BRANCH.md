# Paper Branch `P1` — `p1-phys-spec-ocsr-bench`

> **PERMANENT PAPER BRANCH POLICY (`DO NOT MERGE INTO main`)**
>
> This branch (`p1-phys-spec-ocsr-bench`) is the dedicated, self-contained project branch for **Paper P1** inside `ouroboros-ocsr`.
> - **`main`** remains the clean, reusable core OCSR engine and continues evolving independently.
> - **`p1-phys-spec-ocsr-bench`** permanently preserves Paper P1's specifications, benchmarks, Bayesian physics/NMR disambiguation modules, CLI scripts, and tests without polluting `main` or other `pN-*` paper branches.
> - When citing or archiving this paper, pin to immutable Git tags (`p1-v0.1`, and `p1-v1.0` upon journal submission).

## Paper Metadata

- **Paper ID**: `P1`
- **Working Title**: *Symmetry-Complete Bayesian Physics and Spectroscopy Disambiguation for Stereochemical OCSR and Multimodal Scientific Reasoning (`ChemVeritas-Bench`)*
- **Target Journals**: *Chemical Science* / *JACS Au* / *Digital Discovery* / *J. Chem. Inf. Model. (JCIM)*
- **Spec & Literature Review**: `specs/001-phys-spec-ocsr-bench/` (`42` verified 2019–2026 papers in `research.md`)
- **Packages & Scripts**:
  - `ouroboros/phys/` (`hrms.py`, `epimers.py`, `strain.py`, `nmr.py`, `posterior.py`)
  - `ouroboros/si/` (`parser.py`)
  - `ouroboros/bench/` (`catalog.py`, `builder.py`, `scoring.py`, `vlm_harness.py`)
  - `scripts/build_phys_bench.py`, `scripts/rerank_phys_spec.py`, `scripts/eval_chemveritas.py`, `scripts/extract_si_bench.py`

## Publication Readiness Tracker

- [x] Literature synthesis (`42` papers) & formal hypotheses (`H1.1`–`H1.3`)
- [x] Full Bayesian `C8 × A4` + 3D strain + HRMS + unassigned `13C`/`1H` NMR + Karplus `3J_HH` + chiral abstention engine
- [x] `ChemVeritas-Bench` `Gold-600` (`2,400` visual instances across `V0`–`V3`) generator & `Veritas-RPF` rubric scorer
- [x] Unit & integration test suite (`13/13` passing on CPU)
- [ ] Run full `gold600` ablation matrix (`B0`–`B5`) with `MACE-OFF23-medium` strain backend + real Ouroboros / MolScribe / DECIMER checkpoints
- [ ] Run real frontier VLM evaluations (`Claude` / `GPT` / `Gemini` APIs) across `T1`–`T5`
- [ ] Curate Tier-2 real open-access Supporting Information validation set (`N = 100–150` real SI figure + NMR/HRMS crops)
- [ ] Draft full LaTeX manuscript & publication figures (`papers/p1/`)
