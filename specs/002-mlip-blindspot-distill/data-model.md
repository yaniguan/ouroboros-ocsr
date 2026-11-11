# Data Model: `002-mlip-blindspot-distill`

## Core Entities

1. **`BioisostereRecord` (`ouroboros/active/motifs.py`)**:
   - `mol_id: str`, `smiles: str`, `connectivity_key: str`
   - `group: Literal["bioisostere", "control"]`
   - `family: str` (`bcp_111`, `bch_211`, `oxa_bch_211`, `bchep_311`, `cubane`, `azaspiro_oxetane`, `sf5_polyfluoro`, or `planar_control`)
   - `paired_id: str | None`, `n_heavy_atoms: int`, `eligible: bool`, `rejection_reasons: tuple[str, ...]`

2. **`ConformerFrame` (`ouroboros/active/torsion.py`)**:
   - `frame_id: str`, `mol_id: str`, `smiles: str`, `connectivity_key: str`
   - `group: str`, `family: str`, `source: Literal["torsion", "displaced", "equilibrium"]`
   - `dihedral_deg: float | None`, `transannular_c1_c3_a: float | None`
   - `atomic_numbers: np.ndarray` (`(N,)`, `int64`), `positions: np.ndarray` (`(N, 3)`, `float64`, Å)

3. **`SofteningDiagnostics` (`ouroboros/active/torsion.py`)**:
   - `mol_id: str`, `family: str`
   - `softening_slope_alpha: float` (OLS slope of $\Delta E_{\text{MLIP}}$ vs $\Delta E_{\text{ref}}$; $\alpha < 1$ indicates systematic underprediction of strained barriers)
   - `barrier_height_error_kcal: float`, `rel_energy_mae_kcal: float`, `force_rmse_kcal_a: float`

4. **`CommitteeFrameScore` (`ouroboros/active/committee.py`)**:
   - `frame_id: str`, `mol_id: str`
   - `mean_energy_kcal: float`, `rel_energy_std_kcal: float` (computed *after* per-molecule mean offset subtraction!)
   - `sigma_f_max: float` ($\max_i \sqrt{\frac{1}{M}\sum_m \|\mathbf{F}_i^{(m)} - \bar{\mathbf{F}}_i\|^2}$, kcal/mol/Å)
   - `sigma_f_mean: float`, `env_descriptor: np.ndarray` (`(D,)` LCMD feature vector)

5. **`QMOracleSpec` & `QMLabel` (`ouroboros/active/oracle.py`)**:
   - `QMOracleSpec(backend, method="wb97m-d3bj", basis="def2-tzvppd", solvent="gas", charge=0, multiplicity=1)`
   - `QMLabel(frame_id, energy_kcal, forces_kcal_a, provenance_tag)` — raises `ValueError` if heterogeneous `provenance_tag` labels are mixed in single-head distillation.
