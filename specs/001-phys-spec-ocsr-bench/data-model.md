# Data Model: `001-phys-spec-ocsr-bench`

## 1. `SICharacterization` (`ouroboros/si/parser.py` & `ouroboros/phys/`)

| Field | Type | Description |
|---|---|---|
| `compound_id` | `str` | Compound identifier in scheme/SI (e.g., `"3a"`, `"CV-S1-001"`). |
| `molecular_formula` | `str \| None` | Reported Hill formula if given in prose, else `None`. |
| `hrms_obs_mz` | `float \| None` | Experimental high-resolution $m/z$ value. |
| `hrms_adduct` | `str \| None` | Reported adduct (e.g., `"[M+H]+"`, `"[M+Na]+"`), or `None` to search all standard adducts. |
| `hrms_calcd_mz` | `float \| None` | Literature calculated $m/z$ (used for SI self-consistency check). |
| `c13_peaks_ppm` | `list[float]` | Unassigned $^{13}\text{C}$ NMR chemical shifts in ppm (solvent peaks stripped). |
| `dept_counts` | `dict[str, int] \| None` | Optional DEPT counts (`{"C": int, "CH": int, "CH2": int, "CH3": int}`). |
| `h1_multiplets` | `list[H1Multiplet]` | Unassigned $^1\text{H}$ NMR signals: `(ppm: float, integration: int, multiplicity: str, j_hz: tuple[float, ...])`. |
| `optical_rotation_deg` | `float \| None` | Specific rotation $[\alpha]_D$ when reported (`None` in achiral characterization). |

## 2. `CandidateScore` (`ouroboros/phys/posterior.py`)

| Field | Type | Description |
|---|---|---|
| `raw_smiles` | `str` | Original candidate SMILES (possibly containing superatom aliases). |
| `canonical_smiles` | `str \| None` | Standardized isomeric SMILES after alias expansion (`ouroboros.eval.standardize`). |
| `valid` | `bool` | Whether RDKit standardization succeeded. |
| `visual_logprob` | `float` | Length-normalized visual score $\overline{\log P}_{\text{OCSR}}(M \mid I)$. |
| `hrms_ppm` | `float \| None` | Minimum mass error in ppm across permitted adducts. |
| `hrms_adduct_matched` | `str \| None` | Best-matching ionization adduct. |
| `strain_energy_kcal` | `float \| None` | Continuous excess 3D strain energy (kcal/mol) + clash/CIP-flip penalty. |
| `min_energy_kcal` | `float \| None` | Lowest conformer 3D potential energy (kcal/mol) for relative isomer comparison. |
| `cip_preserved` | `bool` | Whether 3D conformer geometry preserves all specified CIP labels (`stereo_matches`). |
| `min_nonbonded_dist_a` | `float \| None` | Minimum nonbonded interatomic distance (Å) in best conformer. |
| `c13_sym_count` | `int \| None` | Canonical topological $^{13}\text{C}$ symmetry signal count. |
| `c13_mae_ppm` | `float \| None` | Optimal unassigned bipartite $^{13}\text{C}$ shift MAE (ppm). |
| `h1_mae_ppm` | `float \| None` | Optimal unassigned bipartite $^1\text{H}$ shift MAE (ppm). |
| `j_hh_mae_hz` | `float \| None` | Vicinal Karplus $^3J_{\text{HH}}$ coupling discrepancy (Hz). |
| `total_log_posterior` | `float` | Unnormalized joint log-posterior under active `PosteriorWeights`. |
| `posterior_prob` | `float` | Softmax-normalized posterior probability over deduplicated candidate pool. |

## 3. `ChemVeritas` JSONL Task Schema (`ouroboros/bench/builder.py`)

Each molecule record in `molecules.jsonl` stores:
- `mol_id`, `stratum` (`S1_bridged_caged` .. `S5_medchem_bioisosteres`), `truth_smiles`, `flat_smiles`, `inchikey`, `inchikey14`
- `views`: mapping from `V0_clean`, `V1_rotated_offgrid`, `V2_crowded_si_scan`, `V3_condensed_layout` to relative image path and rendering metadata
- `si`: serialized `SICharacterization` + natural-language ACS SI paragraph (`si_text`)
- `hard_negatives`: list of `{role, smiles, category, min_energy_kcal}` covering `enantiomer`, `diastereomer_epimer`, `diastereomer_strained`, `constitutional_isomer`, `near_mass_decoy`
- `veritas_rpf_target`: ground-truth 4-step deduction checkpoints (`formula`, `hrms_adduct_mz`, `c13_symmetry_signals`, `lowest_strain_diastereomer`, `truth_smiles`)
