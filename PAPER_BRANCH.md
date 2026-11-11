# Paper Branch `P2` — `p2-mlip-blindspot-distill`

> **PERMANENT PAPER BRANCH POLICY (`DO NOT MERGE INTO main`)**
>
> This branch (`p2-mlip-blindspot-distill`) is the dedicated, self-contained project branch for **Paper P2** inside `ouroboros-ocsr`.
> - **`main`** remains the clean, reusable core OCSR engine and continues evolving independently.
> - **`p2-mlip-blindspot-distill`** permanently preserves Paper P2's specifications, 7-family bioisostere panel, PES softening diagnostics, active committee selectors, QM oracle adapters, conservative shallow-ensemble distillation trainer, CLI scripts, and tests without polluting `main` or other `pN-*` paper branches.
> - When citing or archiving this paper, pin to immutable Git tags (`p2-v0.1`, and `p2-v1.0` upon journal submission).

## Paper Metadata

- **Paper ID**: `P2`
- **Working Title**: *Literature-Driven Blind-Spot Discovery, PES Softening Diagnostics, and Replay-Protected Conservative Shallow-Ensemble Active Distillation for Universal Organic MLIPs on Modern 3D Bioisosteres*
- **Target Journals**: *J. Chem. Theory Comput. (JCTC)* / *npj Computational Materials* / *Digital Discovery*
- **Spec & Literature Review**: `specs/002-mlip-blindspot-distill/` (`42` verified 2019–2026 papers in `research.md`)
- **Packages & Scripts**:
  - `ouroboros/active/` (`motifs.py`, `torsion.py`, `committee.py`, `oracle.py`, `distill.py`)
  - `scripts/scan_mlip_blindspots.py`, `scripts/run_active_distill.py`

## Publication Readiness Tracker

- [x] Literature synthesis (`42` papers) & formal hypotheses (`H2.1`–`H2.3`)
- [x] `MACE-OFF23` (10-element neutral singlet) vs `AIMNet2` eligibility gates + `112`-scaffold paired 7-family bioisostere panel (`BCP`, `BCH`, `2-oxa-BCH`, `BCHep`, `cubane`, `azaspiro/oxetane`, `-SF5`)
- [x] Bridgehead exit-vector torsion scanner, transannular `C1...C3` compression tracker & PES Softening Slope `alpha`
- [x] Offset-free committee force disagreement `sigma_F` + 5 active selectors (`qbc_lcmd_diverse` et al.)
- [x] Provenance-guarded QM oracle (`psi4` / `pyscf` / `tblite` / `ase` + analytical reference) & conservative `K`-head shallow-ensemble distillation (`8/8` tests passing on CPU)
- [ ] Run full `112`-scaffold blind-spot scan with pretrained `MACE-OFF23` (`small`/`medium`/`large`) and `AIMNet2` weights
- [ ] Compute real DFT single-point energies & gradients (`wB97M-D3(BJ)/def2-TZVPPD` via `PySCF` / `Psi4`, `N = 200-500` active frames + held-out test torsion scans)
- [ ] Run short NVT MD trajectory stability validation before vs after active distillation
- [ ] Draft full LaTeX manuscript & publication figures (`papers/p2/`)
