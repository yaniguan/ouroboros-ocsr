# Tasks: Universal MLIP Blind-Spot Discovery & Active Distillation

**Input**: Design documents from `/specs/002-mlip-blindspot-distill/`

## Phase 1: Setup & Spec-Kit Foundation

- [x] T001 Initialize `.specify/memory/constitution.md` and `.agents/skills/speckit-*/SKILL.md`
- [x] T002 Write 42-paper verified literature synthesis in `specs/002-mlip-blindspot-distill/research.md`
- [x] T003 Author `spec.md`, `plan.md`, and `data-model.md` in `specs/002-mlip-blindspot-distill/`

---

## Phase 2: User Story 1 — Eligibility Gate, 7 Bioisostere Families & Torsion/Softening Scanner (Priority: P1)

- [x] T004 [P] [US1] Implement `MACE_OFF23_POLICY` (10 elements, neutral singlet), superatom/salt standardization, 7 bioisostere SMARTS detectors & matched bioisostere/control library in `ouroboros/active/motifs.py`
- [x] T005 [US1] Implement exit-vector dihedral torsion scanner, off-equilibrium `displaced_conformers` wrapper, transannular $\text{C}_1\cdots\text{C}_3$ tracker & PES Softening Slope $\alpha$ in `ouroboros/active/torsion.py`
- [x] T006 [US1] Add unit tests in `tests/test_active_motifs_torsion.py`

---

## Phase 3: User Story 2 — Committee Force Disagreement, LCMD Active Selector & QM Oracle (Priority: P1)

- [x] T007 [P] [US2] Implement offset-free committee uncertainty evaluator & 5 batch selectors (`random`, `max_displacement`, `energy_std`, `qbc_force`, `qbc_lcmd_diverse`) in `ouroboros/active/committee.py`
- [x] T008 [P] [US2] Implement provenance-guarded `QMOracleSpec` (`ωB97M-D3(BJ)/def2-TZVPPD`), `psi4`/`pyscf`/`tblite` drivers & analytical strained-cage/hypervalent physics reference + baseline committee in `ouroboros/active/oracle.py`
- [x] T009 [US2] Add CLI `scripts/scan_mlip_blindspots.py` and unit tests in `tests/test_active_committee_sampler.py`

---

## Phase 4: User Story 3 — Replay-Protected Conservative Shallow-Ensemble Distillation (Priority: P2)

- [x] T010 [US3] Implement connectivity-disjoint dataset builder, $E(3)$-invariant $K$-head conservative shallow-ensemble $\Delta$-potential ($\mathbf{F} = -\nabla_{\mathbf{R}} E$), replay-regularized trainer & active learning curves in `ouroboros/active/distill.py` and `ouroboros/active/__init__.py`
- [x] T011 [US3] Add CLI `scripts/run_active_distill.py` and end-to-end tests (`H2.1`–`H2.3`, conservative autograd forces & $SO(3)$ rotation equivariance) in `tests/test_active_oracle_distill.py`

---

## Phase 5: Verification & Convergence Gates

- [x] T012 Run `pytest -q`, `ruff check`, and `python scripts/check_no_data.py`; mark all tasks complete and open PR
