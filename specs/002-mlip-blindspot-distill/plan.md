# Implementation Plan: Universal MLIP Blind-Spot Discovery & Active Distillation

**Branch**: `002-mlip-blindspot-distill` | **Date**: 2026-11-11 | **Spec**: `specs/002-mlip-blindspot-distill/spec.md`

## Summary

Implements `ouroboros/active/` and two reproducible CLIs (`scripts/scan_mlip_blindspots.py` and `scripts/run_active_distill.py`), strictly file-disjoint from `main`, PRs `#28`–`#32`, and PR `#33` (`001-phys-spec-ocsr-bench`). Reuses `EligibilityPolicy` (`ouroboros.geometry.eligibility`), `displaced_conformers` (`ouroboros.geometry.displacements`), and `connectivity_key` / `assign_split` (`ouroboros.geometry.splits`).

## Architecture & Module Layout

```text
ouroboros/active/
├── __init__.py          # Public exports
├── motifs.py            # MACE_OFF23_POLICY (10 elements, neutral singlet), 7 bioisostere SMARTS, matched controls
├── torsion.py           # Exit-vector dihedral scanner, transannular C1...C3 distance, PES softening slope alpha
├── committee.py         # Offset-free committee UQ (sigma_F, sigma_dE) + 5 batch selectors incl. qbc_lcmd_diverse
├── oracle.py            # Provenance-guarded QMOracleSpec (wb97m-d3bj/def2-tzvppd; psi4, pyscf, tblite, analytical_cage)
└── distill.py           # Connectivity-split E(3)-invariant K-head conservative shallow-ensemble delta-MLIP + replay buffer
scripts/
├── scan_mlip_blindspots.py
└── run_active_distill.py
tests/
├── test_active_motifs_torsion.py
├── test_active_committee_sampler.py
└── test_active_oracle_distill.py
```

## Constitution Check

- **Principle I (Honest Stratification)**: Out-of-distribution bioisosteres and heavy-atom-matched in-distribution controls are reported separately with counts, Mann-Whitney $U$ $p$-values, and bootstrap 95% CIs.
- **Principle II (Zero Data in Git)**: All torsion scan frames, JSONLs, and checkpoints write only to user/temporary output directories (`scripts/check_no_data.py` clean).
- **Principle III (Connectivity-Disjoint Splits)**: `ouroboros.geometry.splits.assign_split` keeps all conformers, torsions, rattles, and stereoisomers of any connectivity scaffold in one split.
- **Principle IV (Physical Symmetry & Conservation)**: Potential energy is strictly $E(3)$-invariant (pairwise distances $r_{ij}$ + bond/nonbonded cosine angles $\cos\theta_{ijk}$), and atomic forces are exact conservative autograd derivatives $\mathbf{F}_i = -\partial E / \partial \mathbf{r}_i$ (transforming equivariantly under $O(3)$).
- **Principle V (Dual-Track Execution)**: Deterministic CPU analytical reference potential & baseline committee run out-of-the-box in `<5s`; `MACE-OFF23`, `AIMNet2`, `Psi4`, `PySCF`, and `tblite` plug in seamlessly when installed.
