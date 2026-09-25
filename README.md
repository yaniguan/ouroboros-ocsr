# Ouroboros — rotation-equivariant OCSR with stereochemistry → 3D energy

Optical Chemical Structure Recognition (OCSR) pipeline:

```
2D structure drawing (image) ──► SMILES with stereo ──► ETKDG 3D conformers ──► MACE-OFF energy / forces
        image encoder + shared decoder                  RDKit                    E(3)-equivariant MLIP
```

## Research question

Does a rotation-equivariant vision encoder (steerable CNN over C_N and/or group-equivariant
attention) outperform a standard encoder trained with rotation data augmentation, measured by

1. **sample efficiency** — exact-match accuracy vs. training-set size (10k, 50k, 200k, 1M), and
2. **rotation robustness** — accuracy vs. input rotation angle (0–360°, on-grid vs. off-grid angles),

while still recognizing **chirality** correctly (stereo-aware exact match, per-stereocenter accuracy)?

Only the encoder changes between experimental arms; the SMILES decoder is identical.

| Arm | Encoder | Rotation augmentation |
|-----|---------|-----------------------|
| A   | standard CNN baseline | no |
| B   | standard CNN baseline | yes |
| C   | steerable CNN, C_N (N ∈ {4, 8, 16}) | no |
| C+  | steerable CNN, C_N | yes |
| D   | steerable stem + group-equivariant self-attention | no |
| E (optional) | canonicalization network + standard encoder | no |

## Symmetry per stage

| Stage | Input → output | Symmetry group | Property |
|-------|----------------|----------------|----------|
| Image encoder (equivariant arms) | image → feature maps | C_N ⊂ SE(2), discrete rotations only | equivariant (exact for on-grid 90° rotations of the pixel grid; approximate for other angles) |
| Group pooling + relative-position token mixer | feature maps → memory tokens | C_N | invariant token *set* (up to permutation) |
| Image encoder (baseline arms) | image → tokens | none (absolute 2D positional embeddings) | not invariant; rotation robustness only via augmentation |
| SMILES decoder (all arms) | tokens → SMILES | none needed | inherits invariance from the token set |
| Conformer generation | SMILES → 3D coordinates | — | stereo preserved from SMILES |
| MACE-OFF | 3D coordinates → energy, forces | E(3) | energy invariant, forces equivariant |

**Why no reflections in the image encoder.** Mirroring a drawing that contains wedge/hash bonds
turns each stereocenter into its opposite configuration. An encoder invariant to reflections
(D_N or O(2)) would therefore map a molecule and its enantiomer to the same output and could not
recognize chirality. The image encoder uses rotation groups C_N only.

**Why stereo is not evaluated through energy.** Enantiomers have identical MACE energies, so
energy error cannot detect a wrong chirality. Stereo recognition is scored separately with
stereo-aware exact match and per-stereocenter accuracy; the energy stage reports errors by
category (correct / enantiomer / diastereomer / constitutional).

E(3) equivariance appears **only** in the 3D stage (MACE). The image encoder is not E(3)-equivariant.

## Layout

```
ouroboros/
  data/      molecule filtering, rendering with style randomization, WebDataset shards
  encoder/   baseline CNN, escnn steerable C_N encoders, equivariant attention
  decode/    SMILES tokenizer, shared Transformer decoder
  train/     training loop, resumable checkpoints, YAML configs
  eval/      metrics, rotation sweep
  geometry/  ETKDG conformers, MACE-OFF relaxation, error propagation
configs/     experiment configs (generated from one sweep definition)
notebooks/   Colab notebooks (A100 runs)
tests/       pytest suite
benchmarks/  measured throughput / equivariance numbers and figures
TASK.md      live project status, decisions and deviations
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact pins (tested with Python 3.11)
pip install -e .
pytest -q && ruff check .
```

`escnn` depends on `py3nj`, which needs a Fortran compiler (`apt-get install gfortran`) when no
wheel is available. On Colab, `requirements-colab.txt` is used: it is `requirements.txt` without
`torch`/`triton`/CUDA wheels, so the runtime's preinstalled CUDA build of PyTorch is kept.

## Related work (TODO: write up and position against)

- [ ] DECIMER (Rajan et al.) — transformer-based OCSR, image-to-SMILES
- [ ] MolScribe (Qian et al., 2023) — atom/bond graph prediction with coordinates, handles stereo via drawing
- [ ] MolNexTR (Chen et al., 2024) — CNN+ViT dual-stream encoder
- [ ] MolSight — recent OCSR system (TODO: read paper; check its stereochemistry handling and benchmarks)
- [ ] Steerable CNNs / escnn (Weiler & Cesa, 2019); group-equivariant attention (Romero et al.)
- [ ] MACE / MACE-OFF (Batatia et al.; Kovács et al.)
