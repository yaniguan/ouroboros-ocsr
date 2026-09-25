# Ouroboros — rotation-equivariant OCSR with stereochemistry → 3D energy

Optical Chemical Structure Recognition (OCSR) pipeline:

```
2D structure drawing (image) ──► SMILES with stereo ──► ETKDG 3D conformers ──► MACE-OFF energy / forces
        image encoder + shared decoder                  RDKit                    E(3)-equivariant MLIP
```

## Research question

> Does an architectural symmetry prior (SE(2)-equivariant vision encoder) narrow the
> synthetic-to-real gap in OCSR, or — like augmentation-based substitutes — does it help only
> under the rendered conditions it models? And how does it interact with real-data supervision?

Motivation: arXiv:2608.09100 found that synthetic scale, renderer diversity, degradation
augmentation and a verifiable RL objective are lower-cost substitutes for real labeled depictions,
but representative real supervision is what closes the gap on real documents, and that large
real-document gains were obtained with a frozen vision encoder (locating the bottleneck in
supervision rather than visual representation). Ouroboros changes only the vision encoder, so it
tests a different kind of intervention — a built-in symmetry prior — against that finding.

To our knowledge, the first systematic study of group-equivariant vision encoders for OCSR,
evaluated against both augmentation and real-data supervision.

### Pre-registered hypotheses (registered 2026-09-25, before any training result existed)

These are not edited after results arrive; outcomes are reported in a separate results section.

- **H1**: With synthetic-only training, equivariant encoders beat augmentation on rendered and
  rotated test sets.
- **H2**: With synthetic-only training, the equivariant advantage on real-document test sets is
  smaller than on rendered sets, and may be zero.
- **H3**: When real data is mixed in, the equivariant advantage on real documents either persists
  (complementary) or vanishes (subsumed). Either outcome is reported.

Every outcome, including a null result, is a valid result. Metrics, test sets and reporting rules
are fixed in advance (below); they are not tuned after seeing results.

### Measurements

1. exact match (full stereochemistry) on **rendered** and **real-document** test sets, always
   reported as separate columns with per-set counts and bootstrap 95% CIs — never averaged
   together;
2. rotation robustness — accuracy vs. input rotation angle (0–360°, 15° steps, on-grid vs.
   off-grid angles) on both kinds of test set;
3. stereo recognition — stereo-aware exact match and per-stereocenter accuracy;
4. dependence on training-set size and on the fraction of real training data.

Only the encoder changes between experimental arms; the SMILES decoder is identical.

| Arm | Encoder | Rotation augmentation |
|-----|---------|-----------------------|
| A   | standard CNN baseline | no |
| B   | standard CNN baseline | yes |
| C   | steerable CNN, C_N (N ∈ {4, 8, 16}) | no |
| C+  | steerable CNN, C_N | yes |
| D   | steerable stem + group-equivariant self-attention | no |
| E (optional) | canonicalization network + standard encoder | no |

Each arm is crossed with the fraction of real labeled depictions in the training mixture.

## Geometry / MACE stage

The 3D stage (ETKDG conformers → MACE-OFF relaxation) is an **error-propagation analysis**, not a
novel pipeline: it measures how recognition errors (enantiomer, diastereomer, constitutional)
propagate into downstream 3D energies, relative to the energy of the ground-truth molecule.

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

## Pipeline commands

```bash
# data (Phase 1): pool -> stereo-controlled composition -> shards (200k/50k/10k = prefixes of 1M)
python scripts/build_dataset.py --out data/full --preset 1m
python scripts/verify_data.py {splits,labels,mirror,sheet,render-speed,loader-speed} --root data/full
# real documents (Am1-A): manifest (image, smiles, source, split) -> shards; leakage check
python scripts/ingest_real.py --manifest M.csv --out real/<name>
python scripts/check_leakage.py --eval uspto=real/uspto:test --train synth=data/full/manifests/train.tsv
# experiment grid (Phase 4 / Am1-E): one sweep file -> one config per run
python scripts/make_sweep.py && python scripts/estimate_compute.py
# train (resumes automatically from <out>/ckpt/last.pt), evaluate, aggregate
python -m ouroboros.train --config configs/sweep/A_n200k_f0_s0.yaml --out runs/A_n200k_f0_s0
python scripts/evaluate.py --run runs/A_n200k_f0_s0
python scripts/aggregate.py --runs runs/* --out results/
# 3D stage (Phase 6): energies of predicted vs true molecules by error category
python scripts/error_propagation.py --predictions runs/X/eval/predictions.jsonl --out results/X_energy
```

Colab notebooks (`notebooks/`, generated by `scripts/make_colab_nbs.py`): `00_colab_setup`,
`01_generate_data`, `02_train_eval`. Data handling rules: `DATA.md`. Live status: `TASK.md`.

## Related work (TODO: write up and position against)

Motivating prior work:
- [ ] arXiv:2608.09100 — "Real Data Closes Synthetic-to-Real Gap in Optical Chemical Structure
  Recognition": real labeled depictions, not synthetic scale/augmentation, close the gap on real
  documents; gains with a frozen vision encoder.
- [ ] VERDICT (Guan, 2026; arXiv:2608.22183) — companion paper on real-document OCSR.

OCSR systems:
- [ ] DECIMER (Rajan et al.) — transformer-based image-to-SMILES
- [ ] MolScribe (Qian et al., 2023) — atom/bond graph prediction with coordinates
- [ ] MolNexTR (Chen et al., 2024)
- [ ] MolGrapher
- [ ] MolSight
- [ ] MolParser

Adjacent tasks:
- [ ] DeepMoLM — image + 3D-geometry grounding (adjacent but different task)
- [ ] Auto3D — SMILES → 3D with a neural potential (adjacent to the geometry stage)

Methods used here:
- [ ] Steerable CNNs / escnn (Weiler & Cesa, 2019); group-equivariant self-attention (Romero et al.)
- [ ] MACE / MACE-OFF (Batatia et al.; Kovács et al.)
