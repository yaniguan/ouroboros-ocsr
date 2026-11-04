<p>
  <a href="https://antigravity.google/"><img src="assets/google-antigravity-logo.svg" alt="Google Antigravity" height="40" align="left"></a>
  <sub>This project uses API credits from the Google Antigravity for Science program. Thank you, Google Antigravity!</sub>
</p>
<br clear="left">

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ouroboros-ocsr_mark_dark.svg">
  <img src="assets/ouroboros-ocsr_mark_light.svg" alt="Ouroboros logo" width="96">
</picture>

# Ouroboros

**Read a molecule drawing, get the SMILES back, at any rotation.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](pyproject.toml)
[![tests](https://github.com/yaniguan/ouroboros-ocsr/actions/workflows/tests.yml/badge.svg)](https://github.com/yaniguan/ouroboros-ocsr/actions/workflows/tests.yml)

</div>

## What it does

Ouroboros is optical chemical structure recognition (OCSR): it turns an image of a molecule into a SMILES string, stereochemistry included. Its main encoder is a C8 steerable CNN, so a rotated drawing gives rotated features by construction instead of by data augmentation. A fine-tuned ensemble now reads **54.0%** of 298 real figures from papers and patents exactly.

## Quickstart

Python 3.11 and a Fortran compiler (`gfortran`) are needed for `escnn`. No GPU or dataset needed.

```bash
git clone https://github.com/yaniguan/ouroboros-ocsr.git && cd ouroboros-ocsr
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-inference.txt
python scripts/prepare_inference.py
pip install --no-deps -e .
python -m ouroboros.weights fetch --arm C          # downloads + SHA256-checks checkpoints/ouroboros-C.pt
```

```python
from ouroboros.demo.rotation import RotationLab

lab = RotationLab.from_checkpoint("checkpoints/ouroboros-C.pt")    # C8 steerable encoder
for angle in (0, 90, 180):
    r = lab.evaluate("CC(=O)Oc1ccccc1C(=O)O", angle=angle)         # draw aspirin, rotate it
    print(angle, r["rotated_prediction"]["smiles"], "feature error:", r["feature_error"])
```

```text
0 CC(=O)Oc1ccccc1C(=O)O feature error: 3.1e-16
90 CC(=O)Oc1ccccc1C(=O)O feature error: 8.1e-07
180 CC(=O)Oc1ccccc1C(=O)O feature error: 8.1e-07
```

Want a UI? `python -m ouroboros.demo --checkpoint checkpoints/ouroboros-C.pt` opens a local rotation lab ([guide](docs/rotation-demo.md)).

## Key features

- **Rotation-aware encoder.** C8 steerable CNN (arm C) next to a CNN baseline without (A) and with (B) rotation augmentation, same decoder and budget.
- **Stereo-aware SMILES.** Exact match is scored with full stereochemistry after RDKit standardization.
- **Strong decoding.** Beam search, RDKit-validity reranking, multi-model ensembles and test-time augmentation in [`scripts/predict_real.py`](scripts/predict_real.py).
- **One scoring harness.** Exact match, stereo-free match, InChIKey, Tanimoto and bootstrap CIs in [`scripts/eval_predictions.py`](scripts/eval_predictions.py).
- **Leakage check.** [`scripts/check_leakage.py`](scripts/check_leakage.py) fails if any test molecule's InChIKey appears in a training set.
- **Verified weights.** Release checkpoints are fetched with SHA256 checks.

## Results

Real images: 298 figures (ACS 100, UOB 100, USPTO 98), a seeded sample of public OCSR benchmarks. Exact match with stereochemistry.

| Model | All | ACS | UOB | USPTO | Source |
|---|---|---|---|---|---|
| **A + C ensemble, beam 5 + TTA** | **54.0%** (CI 48.3–59.7) | **40%** | **84%** | **37.8%** | [`benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md`](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md) |
| A + C ensemble, beam 5, no TTA | 52.4% (CI 46.6–58.1) | 37% | 85% | 34.7% | [`benchmarks/finetune-v2-oct30/AC-ensemble-beam5-report.md`](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-report.md) |
| A alone, greedy | 51.0% | 38% | 84% | 30.6% | [`benchmarks/finetune-v2-oct30/A-selected-report.md`](benchmarks/finetune-v2-oct30/A-selected-report.md) |
| C alone, greedy | 37.6% | 20% | 74% | 18.4% | [`benchmarks/finetune-v2-oct30/C-selected-report.md`](benchmarks/finetune-v2-oct30/C-selected-report.md) |

Made by [`colab/finetune_v2.ipynb`](colab/finetune_v2.ipynb) (real-data fine-tune v2 of arms A and C) with `scripts/predict_real.py` and `scripts/eval_predictions.py`. Synthetic rotation results for the v0.1.0-alpha checkpoints: [`docs/v0-results.md`](docs/v0-results.md) and [`benchmarks/v0.1.0-alpha/`](benchmarks/v0.1.0-alpha/).

Published systems report higher numbers on the full benchmark sets with their own scoring; our sample is small (95% CIs of about ±10 points per source) and scoring parity is not verified.

## How it works

```mermaid
flowchart LR
    A["Molecule image"] --> B["Encoder<br/>A/B: CNN · C: C8 steerable CNN"]
    B --> C["Token mixer"]
    C --> D["6-layer Transformer decoder"]
    D --> E["Beam search + ensemble<br/>+ TTA + validity rerank"]
    E --> F["SMILES with stereo"]
    F -. "optional" .-> G["3D conformer → MACE energy"]
```

| Arm | Image encoder | Random rotation in training |
|---|---|---|
| A | ResNet-style CNN + absolute positions + Transformer mixer | No |
| B | Same as A | Yes, uniform 0–360° |
| C | C8 steerable CNN + invariant token-set mixer | No |

## Repo layout

```
ouroboros/    package: data/ (rendering, real data), encoder/, decode/, eval/, geometry/, train/, demo/
scripts/      predict, evaluate, build data, profile, export weights
configs/      training configs (pilot-{A,B,C}-200k.yaml, ...)
colab/        GPU notebooks (fine-tune v2, real-data fine-tunes)
notebooks/    setup, data, training, geometry, rotation demo
benchmarks/   measured results, one folder per experiment
docs/         results, release notes, research plan, roadmap
tests/        pytest suite
```

## Limitations / status

- **Research preview.** The GitHub release (v0.1.0-alpha) ships the synthetic-only A/B/C checkpoints. The fine-tuned v2 checkpoints behind the 54.0% result are not released yet.
- **Weak spots on real figures:** stereochemistry (18% exact on the 49 images with stereo), long patent molecules, and R-groups / abbreviations (ACS).
- On upright real figures arm A beats arm C (51.0% vs 37.6% greedy); the rotation prior pays off on rotated drawings.
- C8 is a discrete rotation prior; rasterization means predictions need not match at every angle.
- Hand-drawn structures are not measured yet. Optional 3D/MACE tools are outside the release scope ([DATA.md](DATA.md)).

Weights and provenance: [MODEL_CARD.md](MODEL_CARD.md). Training protocol: [docs/v0-release.md](docs/v0-release.md). History: [TASK.md](TASK.md), [CHANGELOG.md](CHANGELOG.md).

## Citation

```bibtex
@misc{ouroboros2026,
  title  = {Ouroboros: Rotation-Equivariant Optical Chemical Structure Recognition with Stereochemistry},
  author = {Guan, Yani},
  year   = {2026},
  url    = {https://github.com/yaniguan/ouroboros-ocsr},
  note   = {v0.1.0-alpha research preview}
}
```

## License

Code: MIT, see [LICENSE](LICENSE). Weights and dependency provenance: [MODEL_CARD.md](MODEL_CARD.md).
