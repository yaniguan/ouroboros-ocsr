# Feature Specification: Thermodynamically & Spectroscopically Informed OCSR & ChemVeritas-Bench

**Feature Branch**: `001-phys-spec-ocsr-bench`

**Created**: 2026-11-10

**Status**: Approved & Active

**Input**: Physics-in-the-loop Bayesian disambiguation OCSR combining $C_8$ visual priors, continuous 3D thermodynamic strain, and Supporting Information HRMS + $^{13}\text{C}/^1\text{H}$ NMR inversion, paired with `ChemVeritas-Bench` for frontier multimodal AI4Science evaluation.

## Pre-Registered Scientific Hypotheses *(registered prior to experimental evaluation)*

Grounded in our 42-paper literature audit (`specs/001-phys-spec-ocsr-bench/research.md`), specifically addressing the **17% silent stereochemical corruption rate** exposed in `NMRexp` (Wang et al., *Sci. Data* 2025) and the **2D diastereomer blindness** of `TRACE-Chem` (Liu et al., *Symmetry* 2026):

- **H1.1 (Silent Stereochemical Corruption & 3D Energy Amplification)**: On polycyclic/bridged natural products, strained ring junctions, and multi-stereocenter scaffolds, pure 2D visual OCSR and zero-shot frontier VLMs exhibit a $>15\%$ absolute drop from flat connectivity match (`flat_smiles`) to full stereo-exact match (`standardize`), and silent diastereomeric/constitutional misreads induce median downstream 3D conformer energy errors $|\Delta E| > 2.0\text{ kcal/mol}$.
- **H1.2 (Orthogonal Disambiguation by Continuous 3D Strain & Unassigned Spectroscopy)**: Over candidate pools combining visual N-best beams and targeted uncertain-wedge epimers, continuous 3D strain scoring ($\mathcal{E}_{\text{strain}}$, inspired by McDermott, Houk & Garg, *Science* 2024) eliminates $>80\%$ of catastrophic high-strain/clash stereoisomer hallucinations ($|\Delta E| > 10\text{ kcal/mol}$), while exact HRMS ($\pm 5\text{ ppm}$) + $^{13}\text{C}$ symmetry signal counting + Boltzmann-weighted 3D $^{13}\text{C}/^1\text{H}$ shift & Karplus $^3J_{\text{HH}}$ matching lift stereo-exact Top-1 accuracy toward the candidate-pool oracle ceiling and significantly reduce Expected Calibration Error (ECE) and Area Under the Risk-Coverage Curve (AURC).
- **H1.3 (Symmetry Group Epistemic Theorem — Negative Control)**: When the candidate set consists solely of a chiral molecule and its mirror enantiomer $\{M, \bar{M}\}$, achiral 3D scalar energies ($E(3)$-invariant) and achiral solution NMR/HRMS assign identical physical likelihoods ($\Delta \mathcal{L}_{\text{phys+spec}}(M, \bar{M}) = 0$), proving empirically and group-theoretically why reflection-free $C_N \subset SO(2)$ visual encoders are required for absolute chirality and why calibrated abstention is mandatory when planar wedge pixels are destroyed.
- **H1.4 (Frontier Multimodal VLM Reasoning Gap & Reasoning Path Fidelity)**: End-to-end Chain-of-Thought prompting of frontier VLMs (Claude, GPT, Gemini) on paired `(depiction + SI HRMS/NMR)` inputs suffers frequent arithmetic ppm, $^{13}\text{C}$ symmetry-count, and diastereomeric deduction errors captured by our 4-step `Veritas-RPF` rubric, whereas coupling visual candidates with explicit physics-in-the-loop posterior verification consistently improves both Top-1 accuracy and selective precision-coverage.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Bayesian Physics + Spectroscopy OCSR Reranking & Epimer Repair (Priority: P1)

A computational or synthetic chemist runs OCSR on crowded multi-ring or degraded SI depictions accompanied by routine Supporting Information text (`HRMS` + `13C NMR` + `1H NMR`) and receives a physically valid, spectroscopically consistent 3D-verified stereostructure—or an explicit epistemic abstention when planar chirality cues are obliterated.

**Why this priority**: Core scientific engine of Paper A (`ouroboros/phys/`), directly curing silent stereochemical corruption.

**Independent Test**: Run `scripts/rerank_phys_spec.py` on a JSONL of candidate beams + SI spectra and verify ablation ladder `A0` through `A6` plus the enantiomer-only negative control `C0`.

**Acceptance Scenarios**:

1. **Given** a candidate list containing a low-strain true diastereomer, a severely strained trans-fused bridgehead epimer, and a non-isomeric formula decoy, **When** scored with `BayesianPhysSpecReranker`, **Then** HRMS rejects the formula decoy (`>5 ppm`), 3D strain penalizes the clashing epimer, and Boltzmann 3D NMR selects the true diastereomer at Top-1.
2. **Given** only a chiral molecule $M$ and its exact mirror enantiomer $\bar{M}$ with equal visual log-probabilities and achiral NMR/HRMS, **When** evaluated by the reranker, **Then** their physical+spectroscopic scores are identical within `1e-9` and the posterior triggers `abstain_enantiomer_underdetermined`.

---

### User Story 2 - Deterministic `ChemVeritas-Bench` Generation (`Gold-600` & Smoke Presets) (Priority: P1)

An AI4Science researcher generates the leakage-controlled `ChemVeritas-Bench` multimodal benchmark locally in one command without downloading copyrighted PDFs or committing binary files to Git.

**Why this priority**: Establishes the canonical reproducible benchmark (`600` molecules $\times 4$ views $= 2,400$ instances across 5 hard chemical strata and 5 tasks `T1`–`T5`).

**Independent Test**: Run `scripts/build_phys_bench.py --preset smoke --out <tmp>` and verify deterministic SHA256 manifest generation, `InChIKey[:14]` uniqueness, 4 rendered views per molecule, 5 hard negatives per molecule, and tasks `T1`–`T5`.

**Acceptance Scenarios**:

1. **Given** `--preset gold600` (or `--preset smoke`), **When** `build_phys_bench.py` executes, **Then** every generated molecule standardizes cleanly under `ouroboros.eval.standardize`, has distinct `InChIKey[:14]` across strata as configured, and emits complete HRMS + $^{13}\text{C}/^1\text{H}$ NMR + 3D strain ground truth.

---

### User Story 3 - Frontier Multimodal VLM & OCSR Evaluation Harness with `Veritas-RPF` (Priority: P2)

Model developers at frontier AI labs evaluate Claude, GPT, Gemini, MolScribe, DECIMER, and Ouroboros across tasks `T1`–`T5` with automated scoring of structural accuracy, downstream 3D $|\Delta E|$, `Veritas-RPF` reasoning-path fidelity, ECE, and bootstrap 95% CIs.

**Why this priority**: Makes `ChemVeritas-Bench` turnkey for external LLM/VLM adoption.

**Independent Test**: Export VLM prompts via `scripts/eval_chemveritas.py`, evaluate mock/recorded VLM responses and ablation arms, and verify per-view separated tables with bootstrap 95% CIs and `Veritas-RPF` step scores.

**Acceptance Scenarios**:

1. **Given** model predictions on `T1`–`T5` (including reasoning traces), **When** `eval_chemveritas.py` scores them, **Then** rendered (`V0_clean`) and degraded/rotated (`V1`–`V3`) views are reported in separate columns with bootstrap 95% CIs and `Veritas-RPF` step fidelity scores.

---

### User Story 4 - Open-Access Literature SI Characterization Parser & Tier-2 Builder (Priority: P3)

A researcher points `scripts/extract_si_bench.py` at open-access PMC / ChemRxiv Supporting Information text/markdown and figure crops to extract structured `(compound_id, formula, HRMS, 13C NMR, 1H NMR, optical_rotation)` records and validate internal mass consistency.

**Why this priority**: Enables scaling beyond `ChemVeritas-600` to `5,000+` wild open-access SI compounds.

**Independent Test**: Parse representative synthetic organic chemistry SI experimental paragraphs and verify accurate extraction of adducts, ppm errors, $^{13}\text{C}$ $\delta$ lists, and $^1\text{H}$ multiplet/$J$-coupling peaks.

**Acceptance Scenarios**:

1. **Given** raw ACS/RSC/ChemRxiv-formatted SI experimental paragraphs, **When** `parse_si_document` runs, **Then** compound labels, HRMS calcd/found $m/z$, $^{13}\text{C}$ peaks, $^1\text{H}$ multiplets with $J$ couplings in Hz, and $[\alpha]_D$ signs are structured into `SICharacterization` objects.

---

### Edge Cases

- **Condensed Superatom Abbreviations (`Boc, TBS, Fmoc, Pin, Tf...`)**: Expanded cleanly to full atomic graphs prior to HRMS, 3D conformer embedding, and NMR prediction.
- **Accidental Isochrony (Peak Overlap in $^{13}\text{C}$ NMR)**: Experimental peak count $N_{\text{obs}}$ may be slightly smaller than $N_{13\text{C}}^{\text{sym}}$ ($0 \le N_{\text{sym}} - N_{\text{obs}} \le 2$ in large scaffolds) due to accidental overlap, whereas $N_{\text{obs}} > N_{13\text{C}}^{\text{sym}}$ (after solvent stripping) strongly falsifies overly symmetric *meso*/*$C_2$* candidates; penalty must be asymmetric!
- **ETKDGv3 Embedding Failure on Impossible Bridged Topologies**: Treated as a primary physical signal of impossible 3D geometry (assigned maximum strain penalty rather than crashing).
- **Charged Salts in Raw Predictions**: Standardized via `ouroboros.eval.standardize` neutral parent extraction while checking HRMS against the neutral parent's ionizing adducts.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `ouroboros.phys.hrms` MUST compute exact IUPAC monoisotopic masses and evaluate minimum ppm error across common adducts (`[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`, `[M-H]-`, `[M]+.`).
- **FR-002**: `ouroboros.phys.strain` MUST embed stereo-enforced 3D conformers, compute Boltzmann weights, evaluate continuous excess internal strain energy per heavy atom, detect steric clashes and post-relaxation CIP flips (`stereo_matches`), and support both `MACECalculator` and deterministic CPU force-field backends.
- **FR-003**: `ouroboros.phys.nmr` MUST compute canonical topological $^{13}\text{C}$ symmetry signal counts (`breakTies=False`), DEPT multiplicities, Boltzmann-weighted 3D-sensitive $^{13}\text{C}/^1\text{H}$ shifts, Karplus $^3J_{\text{HH}}$ dihedral couplings, and exact monotone/Hungarian unassigned peak-list matching losses.
- **FR-004**: `ouroboros.phys.epimers` MUST expand 37+ organic superatom abbreviations and perform targeted epimer enumeration on up to $m \le 3$ ambiguous/disputed stereocenters.
- **FR-005**: `ouroboros.phys.posterior` MUST combine visual, HRMS, 3D strain, and NMR terms across ablation presets `A0`–`A6` and emit calibrated posterior probabilities plus epistemic parity abstention flags.
- **FR-006**: `ouroboros.si.parser` MUST parse unstructured experimental SI text blocks into typed HRMS, $^{13}\text{C}$ NMR, $^1\text{H}$ NMR (with $J$ couplings and integrations), formula, and $[\alpha]_D$ records.
- **FR-007**: `ouroboros.bench` MUST deterministically construct `ChemVeritas-600` (`120` molecules $\times 5$ strata, $4$ visual regimes $= 2,400$ instances) and `smoke` (`25` molecules) with 5 typed hard negatives per molecule and tasks `T1`–`T5`.
- **FR-008**: `ouroboros.bench.vlm_harness` & `scoring` MUST format multimodal prompts/schemas for Claude, GPT, and Gemini, import external OCSR CSV/JSONL predictions, compute `Veritas-RPF` 4-step rubric fidelity, ECE, AURC, and separate per-view tables with paired bootstrap 95% CIs.

### Key Entities

- **`SICharacterization`**: Co-located analytical observables (`formula`, `hrms_obs_mz`, `hrms_adduct`, `c13_peaks_ppm`, `h1_multiplets`, `optical_rotation_deg`).
- **`CandidateDiagnostics`**: Per-candidate physical/spectroscopic audit (`visual_logprob`, `hrms_ppm_error`, `strain_kcal_mol`, `max_atom_strain`, `cip_preserved`, `c13_sym_count`, `nmr_c13_mae`, `nmr_h1_mae`, `j_hh_mae`, `posterior_prob`).
- **`BenchmarkInstance` & `BenchmarkTaskItem`**: Paired 2D depiction path, ground-truth SMILES/InChIKey, `SICharacterization`, hard negative pool, and task-specific prompt/rubric payload.

## Success Criteria *(mandatory)*

- **SC-001**: Enantiomeric symmetry invariance (`H1.3` negative control) holds to `< 1e-9` absolute difference across HRMS, 3D strain, and achiral NMR modules on 100% of tested chiral pairs.
- **SC-002**: Full posterior (`A6`) strictly outperforms `A0` (visual only), `A2` (+HRMS only), and `A3` (+3D strain only) on `T2` diastereomer + constitutional disambiguation across `ChemVeritas` test instances.
- **SC-003**: `ChemVeritas-600` catalog generates 600 unique valid stereo/bridged/bioisostere structures with 0 invalid SMILES and deterministic SHA256 manifest digest.
- **SC-004**: 100% pass rate on `pytest -q`, `ruff check .`, and `python scripts/check_no_data.py` in CPU CI.
