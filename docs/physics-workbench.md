# Interactive Physics & Equivariance Workbench (`ouroboros.ui`)

`ouroboros.ui` is a self-contained local web workbench (zero external CDN assets or hosted APIs) that brings together every capability of Ouroboros: visual OCSR decoding, 2D $C_N$ rotation & chiral mirror equivariance probes, interactive 3D equivariant molecule generation with a **10-gate physical compliance audit**, 3D error-energy propagation, force-field dataset construction, and batch evaluation.

## Quickstart

From a configured project environment (Python 3.11–3.12):

```bash
python -m ouroboros.ui --port 8787
# Open http://127.0.0.1:8787/ in your browser
```

CLI flags:
- `--port` (default `8787`, pass `0` to pick an unused loopback port automatically)
- `--host` (default `127.0.0.1`)
- `--device` (`cpu` or `cuda`)
- `--threads` (PyTorch CPU intra-op thread count, default `4`)
- `--token` (optional random URL session prefix for multi-user loopback hosts)
- `--ready-file` (optional JSON file written once the listener is bound)

If `checkpoints/ouroboros-{A,B,C}.pt` are present locally, the workbench automatically unlocks `Ouroboros-A`, `Ouroboros-B`, `Ouroboros-C`, `Ouroboros-AC`, and `Ouroboros-ABC` ensembles; otherwise it falls back cleanly to seeded $C_8$ steerable probes so all symmetry and 3D physics checks still run offline.

---

## Ten-Gate 3D Physical & Symmetry Compliance (`ouroboros/ui/compliance.py`)

Tab 1 (`POST /api/compliance3d`) audits any SMILES string (or top-1/top-$K$ OCSR prediction forwarded from Tab 3) across ten explicit physical and Lie-group symmetry gates using `MMFF94s`, `MACE-OFF23-{small,medium,large}`, or `TorchANI ANI-2x`:

1. **`G1_VALENCE_TOPOLOGY`** — Expands supported bracket abbreviations (`[Boc]`, `[Ph]`, `[OMe]`, etc. when enabled), verifies RDKit Kekulization/valence rules, and enforces a single covalently connected molecular graph without unexpanded dummy atoms (`*`) or counterion fragments.
2. **`G2_DOMAIN_ELIGIBILITY`** — Enforces calculator physical domain (`EligibilityPolicy`): net formal charge $Q = 0$, closed-shell singlet (`radicals = 0`), and atomic numbers $Z$ strictly inside the active potential's supported element set.
3. **`G3_STEREO_SPECIFICATION`** — Inventories Cahn–Ingold–Prelog (`R/S` tetrahedral and `E/Z` double-bond) labels and flags `WARN` when stereogenic centers or bonds are left unspecified.
4. **`G4_ETKDG_3D_EMBEDDING`** — Generates chirality-enforced 3D conformers via `ETKDGv3`, rejecting anti-Bredt or triangle-inequality-impossible cages.
5. **`G5_PAULI_STERIC_CLEARANCE`** — Checks all-atom minimum separation ($d_{\min} \ge 0.75\text{ Å}$) and 1–4+ nonbonded van der Waals contacts ($\ge 1.15\text{ Å}$ involving H, $\ge 1.40\text{ Å}$ heavy–heavy). Includes interactive stress-test controls (`perturb_amplitude_a` and `inject_steric_clash`) that highlight collisions directly on the 3D canvas.
6. **`G6_COVALENT_GEOMETRY`** — Audits every covalent bond length against bond-order-adjusted Pyykkö/RDKit covalent radii ($\max |\Delta R_{\text{cov}}| \le 0.30\text{ Å}$) and reports the full valence bond-angle range.
7. **`G7_3D_CHIRALITY_RETENTION`** — Strips input stereochemical tags and re-perceives `R/S` and `E/Z` purely from 3D Cartesian coordinates (`stereo_from_3d`, `stereo_matches`) both before and after LBFGS relaxation.
8. **`G8_STATIONARY_CONVERGENCE`** — Verifies finite potential energy $E$ and nuclear forces $\mathbf{F}$, LBFGS stationary-point convergence (`fmax <= 0.05 eV/Å`), and 298.15 K Boltzmann conformer weights.
9. **`G9_SE3_EQUIVARIANCE_CONSERVATION`** — Draws a live Haar-random proper rotation $\mathbf{Q} \in SO(3)$ ($\det \mathbf{Q} = +1$) and translation $\mathbf{t} \in \mathbb{R}^3$, checking $|E(\mathbf{R}\mathbf{Q}^\top + \mathbf{t}) - E(\mathbf{R})| < 10^{-4}\text{ eV}$, $\max_i \|\mathbf{F}_i(\mathbf{R}\mathbf{Q}^\top + \mathbf{t}) - \mathbf{F}_i(\mathbf{R})\mathbf{Q}^\top\|_2 < 10^{-4}\text{ eV/Å}$, translational Noether conservation ($\|\sum_i \mathbf{F}_i\|_2 < 10^{-3}\text{ eV/Å}$), and rotational Noether torque conservation ($\|\sum_i (\mathbf{r}_i - \bar{\mathbf{r}}) \times \mathbf{F}_i\|_2 < 10^{-3}\text{ eV}$).
10. **`G10_PARITY_AND_CONSERVATIVE_FORCE`** — Applies 3D spatial reflection ($x \to -x$), verifying scalar parity invariance $|E(-\mathbf{x}) - E(\mathbf{x})| < 10^{-4}\text{ eV}$, mirror force equivariance, 3D chiral enantiomer inversion (`categorize == "enantiomer"` for chiral molecules), and a central finite-difference directional derivative test confirming $\mathbf{F} = -\nabla_{\mathbf{R}} E$ (`< 2e-3 eV/Å`).

---

## Overview of All Six Tabs & JSON Endpoints

| Tab | Endpoint | Capability |
|---|---|---|
| **1. 3D Equivariant Mol & Physics** | `POST /api/compliance3d` | Interactive HTML5 3D ball-and-stick + nuclear force vector viewer, `G1`–`G10` compliance scorecard, Boltzmann conformer switcher, `.xyz`/`.extxyz` export |
| **2. 2D $C_N$ Equivariance & Chirality** | `POST /api/equivariance2d` | Compares `steerable` (Arm C), `equiv_attention` (Arm D), `canonicalized` (Arm E), `baseline` (Arm A), and pretrained checkpoints across rotation sweeps + horizontal mirror chirality non-invariance |
| **3. Visual OCSR Studio** | `POST /api/ocsr` | Upload PNG/JPG or render SMILES with arbitrary rotation, `scan` / `handdrawn` noise, R-group substitution, beam search, TTA (`pad`, `sharpen`), and `--min-margin` confidence gating |
| **4. 3D Error Propagation** | `POST /api/propagation` | Compares exact, enantiomer ($\Delta E = 0$), diastereomer, and constitutional candidates on 3D potentials |
| **5. Force-Field Builder** | `POST /api/forcefield` | Runs eligibility filtering, non-equilibrium `displaced_conformers` sampling, leakage-free `connectivity_key` splits, and `.extxyz` + `manifest.jsonl` generation |
| **6. Evaluation & Benchmarks** | `POST /api/evaluate`, `POST /api/benchmark` | Live batch OCSR scoring with 1,000-bootstrap 95% CIs, selective abstention coverage–precision curves, and `benchmarks/*.json` explorer |

## Testing

```bash
pytest tests/test_ui_workbench.py tests/test_rotation_demo.py -q
```
