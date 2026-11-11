# Tasks: Thermodynamically & Spectroscopically Informed OCSR & ChemVeritas-Bench

**Input**: Design documents from `/specs/001-phys-spec-ocsr-bench/`

## Phase 1: Setup & Spec-Kit Foundation

- [x] T001 Initialize `.specify/memory/constitution.md` and `.agents/skills/speckit-*/SKILL.md`
- [x] T002 Write 42-paper verified literature synthesis in `specs/001-phys-spec-ocsr-bench/research.md`
- [x] T003 Author `spec.md`, `plan.md`, and `data-model.md` in `specs/001-phys-spec-ocsr-bench/`

---

## Phase 2: User Story 1 — Physics & Spectroscopy Bayesian OCSR Engine (Priority: P1)

- [x] T004 [P] [US1] Implement exact monoisotopic mass & adduct ppm likelihood in `ouroboros/phys/hrms.py`
- [x] T005 [P] [US1] Implement 50+ superatom expander, $A_4$ stereo Hamming distance & targeted epimer enumerator in `ouroboros/phys/epimers.py`
- [x] T006 [P] [US1] Implement continuous 3D strain, clash & CIP-flip evaluator in `ouroboros/phys/strain.py`
- [x] T007 [P] [US1] Implement $^{13}\text{C}$ symmetry count, Boltzmann 3D shifts, Karplus $^3J_{\text{HH}}$ & exact monotone DP matching in `ouroboros/phys/nmr.py`
- [x] T008 [US1] Implement joint Bayesian posterior, ablations `A0`–`A6`, parity abstention, ECE & AURC in `ouroboros/phys/posterior.py` and `ouroboros/phys/__init__.py`
- [x] T009 [US1] Add CLI `scripts/rerank_phys_spec.py` and unit/symmetry-invariant tests in `tests/test_phys_hrms_nmr.py` and `tests/test_phys_strain_posterior.py`

---

## Phase 3: User Story 4 — Supporting Information Analytical Parser (Priority: P3, Shared Prerequisite for Tier-2 & Bench)

- [x] T010 [P] [US4] Implement SI HRMS, $^{13}\text{C}$ NMR, $^1\text{H}$ NMR ($J$-couplings) & $[\alpha]_D$ parser + formatter in `ouroboros/si/parser.py` and `ouroboros/si/__init__.py`
- [x] T011 [US4] Add CLI `scripts/extract_si_bench.py` and parser unit tests in `tests/test_si_parser.py`

---

## Phase 4: User Stories 2 & 3 — `ChemVeritas-Bench` (`Gold-600`) & Frontier VLM Harness (Priority: P1/P2)

- [x] T012 [P] [US2] Implement deterministic 600-molecule 5-stratum zero-leakage catalog in `ouroboros/bench/catalog.py`
- [x] T013 [US2] Implement 4-view renderer, 5 hard-negative generator & `T1`–`T5` task builder in `ouroboros/bench/builder.py`
- [x] T014 [P] [US3] Implement Claude/GPT/Gemini multimodal prompt schemas, adapters & `Veritas-RPF` parser in `ouroboros/bench/vlm_harness.py`
- [x] T015 [US3] Implement per-view stratified scoring, `Veritas-RPF`, downstream $|\Delta E|$ & paired bootstrap 95% CIs in `ouroboros/bench/scoring.py` and `ouroboros/bench/__init__.py`
- [x] T016 [US2] [US3] Add CLIs `scripts/build_phys_bench.py` & `scripts/eval_chemveritas.py` and end-to-end tests in `tests/test_chemveritas_bench.py`

---

## Phase 5: Verification & Convergence Gates

- [x] T017 Run `pytest -q`, `ruff check .`, and `python scripts/check_no_data.py`; mark all tasks complete and open PR
