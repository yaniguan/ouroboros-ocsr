# Implementation Plan: Thermodynamically & Spectroscopically Informed OCSR & ChemVeritas-Bench

**Branch**: `001-phys-spec-ocsr-bench` | **Date**: 2026-11-10 | **Spec**: `specs/001-phys-spec-ocsr-bench/spec.md`

## Summary

Implement a modular Bayesian physics-and-spectroscopy disambiguation layer (`ouroboros/phys/`), an organic synthesis Supporting Information analytical parser (`ouroboros/si/`), and the `ChemVeritas-Bench` multimodal AI4Science benchmark & VLM evaluation suite (`ouroboros/bench/`). All components reuse `ouroboros.eval.standardize`, `ouroboros.geometry.conformers`, and `ouroboros.decode.rerank` without modifying core `main` files, guaranteeing conflict-free composability with PRs `#28`–`#32`.

## Technical Context

- **Language/Version**: Python 3.11 (`>=3.11,<3.13`)
- **Primary Dependencies**: `rdkit`, `numpy`, `torch`, `pillow` (core CPU CI); optional `mace-torch`, `ase`, and standard HTTPS `urllib`/`httpx` adapters for `Anthropic Claude`, `OpenAI GPT`, and `Google Gemini` APIs
- **Storage**: Deterministic seed catalog in code + out-of-tree generated benchmark directories (`data/chemveritas_*`, ignored by git) with JSONL + SHA256 receipts
- **Testing**: `pytest -q` + `ruff check .` + `python scripts/check_no_data.py`
- **Target Platform**: Cross-platform CPU (Windows/Linux CI) + Linux CUDA GPU for large-scale MACE-OFF23 & OCSR beam sweeps
- **Performance Goals**: `<50 ms` per candidate on CPU default physics+NMR backend; `<20 s` end-to-end smoke benchmark build & evaluation

## Constitution Check

- [x] **Principle I (Pre-Registered Hypotheses & Epistemic Honesty)**: Hypotheses `H1.1`–`H1.4` fixed in `spec.md`; views `V0`–`V3` reported separately with bootstrap 95% CIs; parity abstention enforced when $|\Delta \log P_{\text{OCSR}}| < \tau_{\text{parity}}$ on enantiomer pairs.
- [x] **Principle II (`check_no_data.py` Compliance)**: Zero `.png`, `.tar`, `.parquet`, or `.pt` committed; benchmark builder writes into user-specified output directory.
- [x] **Principle III (`InChIKey[:14]` Anti-Leakage)**: Catalog enforces connectivity-block tracking and optional `--exclude-keys` filtering compatible with `scripts/check_leakage.py`.
- [x] **Principle IV (Symmetry & Physical Invariants)**: Reflection-free $SO(2)$ visual scores combined with $E(3)$-invariant 3D strain & 3D Boltzmann NMR.
- [x] **Principle V (Dual-Track Library + CLI)**: Pure library packages `ouroboros/{phys,si,bench}` + 4 CLI scripts + deterministic CPU unit/integration tests.

## Project Structure

### Documentation (this feature)

```text
specs/001-phys-spec-ocsr-bench/
├── research.md          # 42-paper verified literature review & gap synthesis
├── spec.md              # Pre-registered hypotheses H1.1-H1.4, user stories, requirements
├── plan.md              # Architecture & constitution compliance check
├── data-model.md        # Schemas for SI observables, diagnostics, and ChemVeritas JSONL
└── tasks.md             # Ordered task list with verification gates
```

### Source Code (repository root)

```text
ouroboros/
├── phys/
│   ├── __init__.py
│   ├── hrms.py          # Exact mass, adduct enumeration, ppm error & likelihood
│   ├── strain.py        # ETKDGv3 + MACE/MMFF94s continuous 3D strain, clash & CIP flip
│   ├── nmr.py           # 13C symmetry count, DEPT, 3D Boltzmann shifts, Karplus 3J_HH, DP matching
│   ├── epimers.py       # 37-superatom alias expansion + uncertain-wedge epimer enumeration
│   └── posterior.py     # Bayesian log-posterior, ablations A0-A6, parity abstention, ECE/AURC
├── si/
│   ├── __init__.py
│   └── parser.py        # Regex/grammar SI parser for HRMS, 13C NMR, 1H NMR, formula, [alpha]_D
└── bench/
    ├── __init__.py
    ├── catalog.py       # Deterministic 600-molecule (5x120) stratified stereo/bridged/medchem builder
    ├── builder.py       # 4-view renderer, 5 hard-negative synthesizer, T1-T5 JSONL writer + SHA256
    ├── vlm_harness.py   # Prompt templates, Veritas-RPF parser, Claude/GPT/Gemini/OCSR adapters
    └── scoring.py       # Per-view stratified metrics, Veritas-RPF, downstream |dE|, bootstrap CIs

scripts/
├── build_phys_bench.py  # Build ChemVeritas-600 or smoke benchmark
├── rerank_phys_spec.py  # Run Bayesian physics+spectroscopy reranking & ablations A0-A6
├── eval_chemveritas.py  # Score VLM / OCSR / reranker predictions with bootstrap 95% CIs
└── extract_si_bench.py  # Parse open-access PMC/ChemRxiv SI text + crops into ChemVeritas schema

tests/
├── test_phys_hrms_nmr.py
├── test_phys_strain_posterior.py
├── test_si_parser.py
└── test_chemveritas_bench.py
```
