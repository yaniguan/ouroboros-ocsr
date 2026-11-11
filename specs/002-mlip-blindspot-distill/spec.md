# Feature Specification: Universal MLIP Blind-Spot Discovery, PES Softening Diagnostics & Replay-Regularized Active Distillation

**Feature Branch**: `002-mlip-blindspot-distill`  
**Created**: 2026-11-11  
**Status**: Approved  
**Target Venues**: *J. Chem. Theory Comput. (JCTC)* / *npj Computational Materials* / *Digital Discovery*

## 1. Scientific Motivation & Literature Positioning (`specs/002-mlip-blindspot-distill/research.md`)

Universal organic machine-learning interatomic potentials (`MACE-OFF23`, Kovács et al., *JACS* 2025; `AIMNet2`, Anstine et al., *Chem. Sci.* 2025; `ANI-2x`, Devereux et al., *JCTC* 2020) are trained on historical molecule repositories (`SPICE` + `QMugs`, curated pre-2021). Meanwhile, contemporary medicinal chemistry (*J. Med. Chem.*, *Nature*, *Nat. Chem.*, and post-2021 patents) has undergone an "Escape from Flatland 2.0" surge of strained 3D saturated bioisosteres and hypervalent fluorinated motifs:
- `BCP` (`bicyclo[1.1.1]pentane`, *para* phenyl bioisostere; nonbonded $\text{C}_1\cdots\text{C}_3 \approx 1.85\text{ \AA}$),
- `BCH` (`bicyclo[2.1.1]hexane`, *ortho*/*meta* benzene bioisostere; Kleinmans et al., *Nature* 2022),
- `2-oxa-BCH` (`2-oxabicyclo[2.1.1]hexane`; Denisenko et al., *Nat. Chem.* 2023),
- `BCHep` (`bicyclo[3.1.1]heptane`, *meta* benzene bioisostere; Frank et al., *Nature* 2022),
- `Cubane` (`1,4-cubane` rigid linker),
- `Azaspiro / Oxetane` (`2-oxa-6-azaspiro[3.3]heptane` & spiro-oxetanes),
- `SF5 / Polyfluoro` (octahedral pentafluorosulfanyl aryl/alkyl exit vectors).

Every single one of these 7 families lies **100% inside** `MACE-OFF23`'s neutral closed-shell 10-element chemical domain (`{H, C, N, O, F, P, S, Cl, Br, I}`, $Q=0, 2S+1=1$; Kovács et al., *JACS* 2025), yet suffers systematic out-of-distribution torsional error and **PES barrier softening** ($\alpha < 1$; extending Deng, Persson & Ceder, *npj Comput. Mater.* 2025 from inorganic crystals to strained organic cages).

## 2. Falsifiable Scientific Hypotheses

- **H2.1 (Strained Bioisostere Blind Spot & PES Softening)**: Post-2020 3D bioisosteres passing `MACE_OFF23_POLICY` exhibit significantly higher committee force disagreement $\sigma_F(\mathbf{R})$, larger dihedral barrier height errors, and systematic PES softening slopes $\alpha < 1$ relative to heavy-atom-matched planar aromatic/unstrained aliphatic controls ($p < 10^{-3}$, Mann-Whitney $U$ + bootstrap 95% CIs).
- **H2.2 (Diversity-Aware Committee Active Selection Efficiency)**: Selecting QM query frames via committee force variance + LCMD feature-diversity (`qbc_lcmd_diverse`; Zaverkin et al., *Digital Discovery* 2022 / Holzmüller et al., *JMLR* 2023) achieves lower held-out conformational $\Delta E$ MAE and force RMSE per single-point QM budget $B$ than `random`, `max_displacement`, `energy_std`, or naive `qbc_force` on connectivity-disjoint scaffolds (`ouroboros.geometry.splits.assign_split`).
- **H2.3 (Forgetting-Free Chemical Accuracy Recovery)**: Replay-regularized conservative $E(3)$-invariant shallow-ensemble $\Delta$-distillation (combining Kaur, Batatia & Csányi, *Faraday Discuss.* 2025 replay protection with Kellner & Ceriotti, *MLST* 2024 shallow readout ensembles) restores held-out bioisostere relative torsion/conformer energy MAE from $>2.0\text{ kcal/mol}$ to chemical accuracy ($<0.5\text{ kcal/mol}$), lifts softening slope $\alpha \to 1.0$, preserves exact conservative forces ($\mathbf{F} = -\nabla_{\mathbf{R}} E$), and incurs $<0.1\text{ kcal/mol}$ drift on in-distribution controls.

## 3. User Stories & Acceptance Scenarios

### User Story 1 — Strict Eligibility & 7-Family Bioisostere Torsion Blind-Spot Scanner (Priority: P1)
Researchers scan SMILES extracted from medicinal-chemistry depictions (`predict_real.py --min-margin`) through `MACE_OFF23_POLICY`, detect 7 modern 3D bioisostere families + matched planar controls, generate bridgehead exit-vector torsion scans ($15^\circ$ grid) + off-equilibrium rattle (`displaced_conformers`), and quantify transannular $\text{C}_1\cdots\text{C}_3$ compression and PES softening slope $\alpha$.

### User Story 2 — Level-of-Theory Guarded Committee Active Selector & QM Oracle (Priority: P1)
Researchers score candidate 3D geometries with a multi-model/multi-head MLIP committee (subtracting per-molecule mean offsets before computing $\sigma_{\Delta E}$), select non-redundant high-uncertainty frames via `qbc_lcmd_diverse`, and label only the selected frames with level-of-theory provenance-locked QM (`ωB97M-D3(BJ)/def2-TZVPPD` via `Psi4`/`PySCF`, or deterministic reference physics backend on CPU).

### User Story 3 — Replay-Protected Conservative Shallow-Ensemble Distillation (Priority: P2)
Researchers split molecules strictly by `connectivity_key` (`train/val/test`), train a conservative $E(3)$-invariant radial-angular $K$-head shallow-ensemble $\Delta$-potential ($\mathbf{F} = -\nabla_{\mathbf{R}} E$) regularized with an in-distribution replay buffer across budgets $B \in \{16, 32, 64, 128\}$, and verify both $<0.5\text{ kcal/mol}$ held-out bioisostere accuracy and calibrated ensemble force UQ.
