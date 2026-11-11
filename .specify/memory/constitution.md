# Ouroboros-OCSR Scientific & Engineering Constitution

## Core Principles

### I. Pre-Registered Scientific Hypotheses & Epistemic Honesty (NON-NEGOTIABLE)
Every research feature and benchmark MUST register falsifiable hypotheses, primary metrics, and subgroup breakdowns in `specs/<feature>/spec.md` before evaluation results are aggregated. Every outcome—including a null result—is a valid scientific result. Rendered and real-document accuracy MUST always be reported in separate columns with counts and bootstrap 95% confidence intervals, never averaged together. When available physical/spectroscopic observables are symmetry-blind to a structural degree of freedom (e.g., achiral NMR and scalar $E(3)$-invariant energies on pure enantiomers $M$ vs. $\bar{M}$), systems MUST preserve epistemic honesty (relying on $SO(2)$ visual parity priors or calibrated abstention rather than fabricating certainty).

### II. Zero-Data Git Hygiene & Deterministic Receipts (`check_no_data.py`)
Raw datasets, WebDataset shards (`.tar`), model checkpoints (`.pt`, `.ckpt`), parquet/NPZ archives, copyrighted PDFs, and raster images outside `benchmarks/` MUST NEVER be committed to Git (`scripts/check_no_data.py` exits 0 on every commit). Benchmark corpora are versioned via compact, deterministic code/JSONL seed specifications (<5 MB) paired with SHA256 generation receipts.

### III. Connectivity-Disjoint Anti-Leakage Protocol
All train/validation/test splits and active-learning holdouts MUST group molecules by flat topological connectivity (`connectivity_key` / 14-character standard InChIKey connectivity block `InChIKey[:14]`) so stereoisomers, conformers, and tautomers of a test scaffold never leak into training or hyperparameter calibration.

### IV. Physical & Quantum-Chemical Invariant Alignment
Every physics or spectroscopy module MUST enforce the domain of validity of its underlying potential or observable:
- Symmetry groups MUST match physical observables ($C_N \subset SO(2)$ without reflections for chiral 2D depictions; $E(3)$-invariant energies and $SO(3)$-equivariant conservative forces $\mathbf{F} = -\nabla_{\mathbf{R}} E$ in 3D).
- Universal MLIP integrations (`MACE-OFF23`) MUST enforce supported element sets (`{H, C, N, O, F, P, S, Cl, Br, I}`), formal charge neutrality (`Q = 0`), and closed-shell singlet spin (`2S+1 = 1`) via `EligibilityPolicy`, and MUST NOT mix incompatible DFT basis/functional absolute energy scales (`ωB97M-D3(BJ)/def2-TZVPPD`) without explicit offset removal or multi-head replay.

### V. Dual-Track Reproducibility, Library-First & CLI Contracts
Every capability lives first as a modular, pure-Python/PyTorch/RDKit library package under `ouroboros/` with a deterministic CPU-runnable default backend (passing `pytest -q` in Linux/Windows CPU CI without network or GPU) and clean pluggable adapters for heavy production backends (`MACE-OFF23`, `Psi4`/`PySCF` DFT, `Claude`/`GPT`/`Gemini` multimodal APIs). Every workflow exposes a CLI under `scripts/` emitting machine-readable JSON/JSONL with provenance metadata.

## Code Quality & Verification Gates

1. **Static Analysis**: `ruff check .` MUST pass with zero warnings (`line-length = 100`, rules `E, F, W, I, B, UP`).
2. **Automated Test Suite**: Every module and scientific invariant (including negative controls such as enantiomeric energy/NMR degeneracy and connectivity-split disjointness) MUST be covered by deterministic `pytest` tests.
3. **Pre-Commit Data Guard**: `python scripts/check_no_data.py` MUST pass with zero violations prior to every commit and pull request.

## Governance

This Constitution governs all `specs/<feature>/` specifications, implementation plans, task lists, and pull requests in `ouroboros-ocsr`. Any architectural exception MUST be explicitly justified in the `Complexity Tracking` table of the feature's `plan.md`.

**Version**: 1.0.0 | **Ratified**: 2026-11-10 | **Last Amended**: 2026-11-10
