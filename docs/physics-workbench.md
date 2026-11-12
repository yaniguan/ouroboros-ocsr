# Molecule Reader & 3D Physics Lab (`ouroboros.ui`)

`ouroboros.ui` is a local browser app with no cloud calls or external assets. It lets you read molecule pictures into SMILES, test 2D rotations and mirror flips, and build 3D shapes while checking 10 basic chemistry and physics rules.

## How to run

Use Python 3.11 or 3.12 from the repo root:

```bash
python -m ouroboros.ui --port 8787
# Open http://127.0.0.1:8787/ in your browser
```

Useful flags:
- `--port` (default `8787`, or `0` to pick any free port)
- `--host` (default `127.0.0.1`)
- `--device` (`cpu` or `cuda`)
- `--threads` (CPU thread count, default `4`)
- `--token` (adds a random URL prefix if you share the machine)

If `checkpoints/ouroboros-{A,B,C}.pt` are in `checkpoints/`, the app uses them automatically. If not, it uses small built-in test networks so all 2D rotation and 3D physics checks still work offline.

## 10 Chemistry & Physics Checks (`ouroboros/ui/compliance.py`)

Tab 1 turns any SMILES string into a 3D molecule using `MMFF94s`, `MACE-OFF23`, or `ANI-2x`, and checks 10 rules:

1. **Valid chemical bonds** — Expands shorthand groups (`[Boc]`, `[Ph]`, `[OMe]`, etc.), checks atom valences, and rejects broken strings or loose salt ions.
2. **Neutral charge & supported elements** — Checks that total charge is 0, all electrons are paired, and the 3D force model supports every element in the molecule.
3. **3D handedness (R/S and E/Z) labeled** — Warns when chiral centers or double bonds are left unlabeled.
4. **Can fold into 3D space** — Builds 3D shapes with RDKit `ETKDGv3` and rejects impossible strained cages.
5. **Atoms do not crash into each other** — Checks that no two atoms overlap (`>= 0.75 A` for any pair, `>= 1.15 A` with H, `>= 1.40 A` between heavy atoms). Includes a "Force an atom crash" toggle to test failure detection on the 3D canvas.
6. **Realistic bond lengths & angles** — Checks every bond length against standard covalent radii (`<= 0.30 A` difference).
7. **3D shape keeps right/left handedness** — Re-reads R/S and E/Z directly from the 3D atom coordinates before and after relaxation to make sure nothing flipped.
8. **3D shape settles to a calm minimum** — Relaxes atom positions until the biggest force on any atom is small (`<= 0.05 eV/A`) and computes room-temperature weights.
9. **Moving or rotating in 3D keeps physics unchanged** — Rotates and shifts the whole molecule in 3D space and checks that energy stays the same, force arrows rotate with the molecule, and internal forces add up to zero net push and zero net twist.
10. **Mirror test & force matches energy slope** — Flips `x -> -x` to check that mirror-twin molecules have the same energy (while R/S flips), and nudges an atom slightly to verify that force equals `-dE/dx`.

## What each tab does

| Tab | API path | What you can try |
|---|---|---|
| **1. 3D Molecule & Physics Check** | `POST /api/compliance3d` | Drag to rotate the 3D ball-and-stick view, toggle green force arrows, inspect all 10 checks, and download `.xyz` or `.extxyz` files. |
| **2. 2D Rotation & Mirror Test** | `POST /api/equivariance2d` | Rotate a 2D drawing from 0° to 315° and see why steerable networks keep matching features while mirror flips stay distinct for chiral molecules. |
| **3. Read Molecule Image** | `POST /api/ocsr` | Upload a PNG/JPG or draw a SMILES at any angle, add scan or hand-drawn noise, read back top candidate SMILES, and send any guess to the 3D lab. |
| **4. Compare Mistakes in 3D** | `POST /api/propagation` | Compare the true molecule against mirror twins (0 energy gap) and wrong 3D shapes (clear energy gap). |
| **5. Build 3D Dataset** | `POST /api/forcefield` | Turn a list of SMILES into `.extxyz` 3D coordinates, energies, and forces with train/val/test group tags. |
| **6. Batch Score** | `POST /api/evaluate`, `POST /api/benchmark` | Score a batch of molecules, inspect confidence filtering, or view saved JSON results from `benchmarks/`. |

## Run tests

```bash
pytest tests/test_ui_workbench.py tests/test_rotation_demo.py -q
```
