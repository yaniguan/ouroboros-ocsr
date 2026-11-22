<p>
  <a href="https://antigravity.google/"><img src="assets/google-antigravity-logo.svg" alt="Google Antigravity" height="40" align="left"></a>
  <sub>This project uses API credits from the Google Antigravity for Science program. Thank you, Google Antigravity!</sub>
</p>
<br clear="left">

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ouroboros-ocsr_mark_dark.svg">
  <img src="assets/ouroboros-ocsr_mark_light.svg" alt="Ouroboros logo" width="84">
</picture>

# Ouroboros

**Read a molecule drawing, get the SMILES back, at any rotation — verified in 3D.**

<p>
  <strong>Project Website:</strong> <a href="https://yaniguan.github.io/ouroboros-ocsr/">https://yaniguan.github.io/ouroboros-ocsr/</a><br>
  <strong>Browser 3D Lab (No Install):</strong> <a href="https://yaniguan.github.io/ouroboros-ocsr/lab/">https://yaniguan.github.io/ouroboros-ocsr/lab/</a>
</p>

[![Project Website](https://img.shields.io/badge/website-yaniguan.github.io%2Fouroboros--ocsr-b5d88e?labelColor=101820)](https://yaniguan.github.io/ouroboros-ocsr/)
[![Browser 3D Lab](https://img.shields.io/badge/Browser_3D_Lab-live_demo-79c2bd?labelColor=101820)](https://yaniguan.github.io/ouroboros-ocsr/lab/)
[![Paper P1](https://img.shields.io/badge/Paper_P1-ChemVeritas--600-b9a1de?labelColor=101820)](https://github.com/yaniguan/ouroboros-ocsr/tree/p1-phys-spec-ocsr-bench)
[![Paper P2](https://img.shields.io/badge/Paper_P2-Bioisostere_MLIP-91b6df?labelColor=101820)](https://github.com/yaniguan/ouroboros-ocsr/tree/p2-mlip-blindspot-distill)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](pyproject.toml)
[![tests](https://github.com/yaniguan/ouroboros-ocsr/actions/workflows/tests.yml/badge.svg)](https://github.com/yaniguan/ouroboros-ocsr/actions/workflows/tests.yml)

</div>

## What it does

Ouroboros turns an image of a molecule into a SMILES string with stereochemistry. On synthetic rotated drawings, its C8 steerable CNN (arm C) matches rotation augmentation without rotating training images ([65.8% vs 65.7%](docs/v0-results.md) across 24 angles). On real figures from papers and patents, the fine-tuned A + C ensemble scores [**46.7%**](benchmarks/frozen-test-v2/AC-ensemble-beam5-ttapad-report.md) (`85.0%` on UOB) on the untouched 300-image `frozen-test-v2` holdout with validation-locked decoding, and [**54.0%**](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md) on the 298-image `subset100` dev-test set (where plain CNN arm A is the stronger single model: [51.0%](benchmarks/finetune-v2-oct30/A-selected-report.md) vs [37.6%](benchmarks/finetune-v2-oct30/C-selected-report.md) for C).

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

Want a UI?
- **No install (GitHub Pages):** open [https://yaniguan.github.io/ouroboros-ocsr/lab/](https://yaniguan.github.io/ouroboros-ocsr/lab/) directly in any browser to inspect 3D conformers, 10 physics checks, C8 steerable feature maps, and custom SMILES.
- **Full local Python server:** `python -m ouroboros.ui --port 8787` opens the local web app with live PyTorch OCSR decoding and MACE-OFF23 relaxation ([guide](docs/physics-workbench.md)).
- `python -m ouroboros.demo --checkpoint checkpoints/ouroboros-C.pt` opens the small 2D rotation demo ([guide](docs/rotation-demo.md)).

## Key features

- **Rotation-aware encoder.** C8 steerable CNN (arm C) next to a CNN baseline without (A) and with (B) rotation augmentation, same decoder and budget.
- **Stereo-aware SMILES.** Exact match is scored with full stereochemistry after RDKit standardization.
- **Strong decoding.** Beam search, RDKit-validity reranking, multi-model ensembles and test-time augmentation in [`scripts/predict_real.py`](scripts/predict_real.py).
- **One scoring harness.** Exact match, stereo-free match, InChIKey, Tanimoto and bootstrap CIs in [`scripts/eval_predictions.py`](scripts/eval_predictions.py).
- **Leakage check.** [`scripts/check_leakage.py`](scripts/check_leakage.py) fails if any test molecule's InChIKey appears in a training set.
- **Verified weights.** Release checkpoints are fetched with SHA256 checks.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-chart-dark.svg">
  <img src="assets/results-chart-light.svg" alt="Measured Ouroboros accuracy across 24 rotation angles and 298 real document figures" width="100%">
</picture>

**Held-out release test (`frozen-test-v2`, scored once):** 300 figures (ACS 100, UOB 100, USPTO 100; 57 `has_stereo`, 243 `no_stereo`), zero molecule overlap with `subset100` or training/validation data. Decoding (`AC-ensemble-beam5-ttapad`) was locked strictly on `PMC val` (`44.6%` across 963 validation images, [`benchmarks/frozen-test-v2/val_selection.json`](benchmarks/frozen-test-v2/val_selection.json)) before scoring `frozen-test-v2` once.

| Model (locked on `PMC val`) | All (`n=300`) | ACS | UOB | USPTO | `has_stereo` (`n=57`) | `no_stereo` (`n=243`) | Source |
|---|---|---|---|---|---|---|---|
| **A + C ensemble, beam 5 + `pad` TTA** | **46.7%** (CI 40.7–51.7) | **30%** | **85%** | **25%** | **1.8%** | **57.2%** | [`benchmarks/frozen-test-v2/AC-ensemble-beam5-ttapad-report.md`](benchmarks/frozen-test-v2/AC-ensemble-beam5-ttapad-report.md) |

**Dev-test set (`subset100`):** 298 figures (ACS 100, UOB 100, USPTO 98; 49 `has_stereo`, 249 `no_stereo`). Exact match with stereochemistry.

| Model | All | ACS | UOB | USPTO | Source |
|---|---|---|---|---|---|
| **A + C ensemble, beam 5 + TTA (`dev-test`)** | **54.0%** (CI 48.3–59.7) | **40%** | **84%** | **37.8%** | [`benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md`](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md) |
| A + C ensemble, beam 5, no TTA | 52.3% (CI 46.6–58.1) | 37% | 85% | 34.7% | [`benchmarks/finetune-v2-oct30/AC-ensemble-beam5-report.md`](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-report.md) |
| A alone, greedy | 51.0% | 38% | 84% | 30.6% | [`benchmarks/finetune-v2-oct30/A-selected-report.md`](benchmarks/finetune-v2-oct30/A-selected-report.md) |
| C alone, greedy | 37.6% | 20% | 74% | 18.4% | [`benchmarks/finetune-v2-oct30/C-selected-report.md`](benchmarks/finetune-v2-oct30/C-selected-report.md) |

Made by [`colab/finetune_v2.ipynb`](colab/finetune_v2.ipynb) (real-data fine-tune v2 of arms A and C) with `scripts/predict_real.py` and `scripts/eval_predictions.py`. Because the 298-image `subset100` set was watched across multiple rounds of model and decoding work, it is kept as a **dev-test** set (`54.0%`), whereas `frozen-test-v2` (`46.7%`) is scored **once per release** after all choices are fixed on validation ([`DATA.md`](DATA.md)).

The Quickstart above downloads the synthetic `v0.1.0-alpha` checkpoint, not the unreleased fine-tuned v2 weights behind the 54.0% model. Synthetic rotation results for the `v0.1.0-alpha` checkpoints: [`docs/v0-results.md`](docs/v0-results.md) and [`benchmarks/v0.1.0-alpha/`](benchmarks/v0.1.0-alpha/).

Published systems report higher numbers on the full benchmark sets with their own scoring; our sample is small (95% CIs of about ±10 points per source) and scoring parity is not verified.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/overview-dark.svg">
  <img src="assets/overview-light.svg" alt="Ouroboros pipeline from rotated 2D depiction through C8 steerable encoder and Transformer decoder to 3D MACE forces and spectra" width="100%">
</picture>

| Arm | Image encoder | Random rotation in training |
|---|---|---|
| A | ResNet-style CNN + absolute positions + Transformer mixer | No |
| B | Same as A | Yes, uniform 0–360° |
| C | C8 steerable CNN + invariant token-set mixer | No |

## Research extensions (frozen paper branches)

- **Paper P1 — [`p1-phys-spec-ocsr-bench`](https://github.com/yaniguan/ouroboros-ocsr/tree/p1-phys-spec-ocsr-bench) (tag `p1-v1.0`):** *Thermodynamically and Spectroscopically Informed OCSR and the `ChemVeritas-600` Multimodal Science Benchmark.* Pairs visual beams with 3D MACE-OFF23 Boltzmann strain (`+45.0 kcal/mol` median penalty on silent stereo flips, `ROC-AUC = 0.853`), exact-mass HRMS, and unassigned ¹³C/¹H NMR matching (`19.7% → 77.0%` T2 stereoisomer accuracy; `73.3%` recovery on 30 published literature ¹³C NMR spectra).
- **Paper P2 — [`p2-mlip-blindspot-distill`](https://github.com/yaniguan/ouroboros-ocsr/tree/p2-mlip-blindspot-distill):** *Foundation Force-Field Blind-Spot Discovery and Replay-Protected Active Distillation on Modern 3D Bioisosteres.* Audits PES barrier softening across 112 paired sp³ cages (`BCP`, `BCH`, `2-oxa-BCH`, `BCHep`, `cubane`, `azaspiro/oxetane`, `-SF₅`) with conservative `K`-head shallow ensembles and anchor replay.
- **Interactive overview:** [https://yaniguan.github.io/ouroboros-ocsr/](https://yaniguan.github.io/ouroboros-ocsr/) (`site/`).

## Repo layout

```
ouroboros/    package: data/ (rendering, real data), encoder/, decode/, eval/, geometry/, train/, demo/, ui/
scripts/      predict, evaluate, build data, profile, export weights
configs/      training configs (pilot-{A,B,C}-200k.yaml, ...)
colab/        GPU notebooks (fine-tune v2, real-data fine-tunes)
notebooks/    setup, data, training, geometry, rotation demo
benchmarks/   measured results, one folder per experiment
site/         project website (https://yaniguan.github.io/ouroboros-ocsr/)
docs/         results, release notes, research plan, roadmap
tests/        pytest suite
```

## Limitations / status

- **Research preview.** The GitHub release (v0.1.0-alpha) ships the synthetic-only A/B/C checkpoints. The fine-tuned v2 checkpoints behind the `46.7%` `frozen-test-v2` and `54.0%` `dev-test` results are not released yet.
- **Weak spots on real figures:** stereochemistry (fine-tune v2 scores [1.8%](benchmarks/frozen-test-v2/AC-ensemble-beam5-ttapad-report.md) on the 57 `has_stereo` figures in `frozen-test-v2` vs `57.2%` on `no_stereo`, and on the 49 `has_stereo` figures in `subset100` scores [18.4%](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-ttapad-report.md) with `pad` TTA, [20.4%](benchmarks/finetune-v2-oct30/AC-ensemble-beam5-report.md) without TTA, [16.3%](benchmarks/finetune-v2-oct30/A-selected-report.md) for arm A greedy, and [8.2%](benchmarks/finetune-v2-oct30/C-selected-report.md) for arm C greedy, vs [2.0%](benchmarks/real-subset100-rgroups/A_report.json) for released arm A), long patent molecules, and R-groups / abbreviations (ACS).
- On upright real figures arm A beats arm C (51.0% vs 37.6% greedy); the rotation prior pays off on rotated drawings.
- C8 is a discrete rotation prior; rasterization means predictions need not match at every angle.
- **Hand-drawn structures:** on a 200-image molecule-disjoint sample of the public DECIMER Hand-drawn set ([`benchmarks/handdrawn-decimer/`](benchmarks/handdrawn-decimer/), `10.5281/zenodo.6456306`), released `v0.1.0-alpha` arms A/B/C score `0.0%` exact match and fine-tune v2 scores `3.0%` (arm A greedy), `2.0%` (arm C greedy), and `4.5%` (CI `2.0–7.5%`, A + C beam 5); neither release trained on hand-drawn data yet. Optional 3D/MACE tools are outside the release scope ([DATA.md](DATA.md)).

Weights and provenance: [MODEL_CARD.md](MODEL_CARD.md). Training protocol: [docs/v0-release.md](docs/v0-release.md). History: [CHANGELOG.md](CHANGELOG.md).

## Optional tools

Use `--ink-crop` with [`scripts/ingest_real.py`](scripts/ingest_real.py) or [`scripts/image_to_forcefield.py`](scripts/image_to_forcefield.py) `--images` to remove white padding before resizing. It keeps all marks darker than `--ink-threshold` (default 250), with `--ink-margin` pixels of padding (default 8, before resizing). Blank images keep their size. Cropping is off by default. Use the same settings when preparing training images and making predictions.

To check ranked candidates, export them with [`scripts/predict_real.py`](scripts/predict_real.py) `--nbest-out candidates.jsonl`, then run [`scripts/eval_nbest.py`](scripts/eval_nbest.py) `candidates.jsonl --k 1 3 5`. The report uses unique standardized molecules. For each subset, it checks for the correct molecule among the top candidates, with and without stereochemistry. It also counts cases where the correct candidate was not selected.
Prediction checks are off by default. [`scripts/predict_real.py`](scripts/predict_real.py) `--min-margin 0.2 --max-disagreement 0` adds acceptance fields and keeps the predicted molecule. The margin is the score gap between distinct molecules after standardization. Missing alternatives, or scores that are missing, infinite, or NaN, leave the margin unknown. Agreement checks keep stereochemistry. They do not give the probability that a prediction is correct. Run [`scripts/eval_selective.py`](scripts/eval_selective.py) `validation.jsonl --margins 0 0.2 0.5` on validation data with ranked candidates to see how many predictions pass and how often they are correct before choosing limits. [`scripts/image_to_forcefield.py`](scripts/image_to_forcefield.py) `--respect-confidence` reads acceptance fields from the predictions CSV. Rows with rejected or missing acceptance get a manifest record without geometry.
[`scripts/image_to_forcefield.py`](scripts/image_to_forcefield.py) `--expand-aliases` expands Me, Et, OMe, OEt, Ph, Boc, CO2Me and CO2Et before making 3D coordinates. Each alias must have one single bond to one other atom. Generic R groups and unknown aliases are still rejected. The manifest saves the input, recognized, expanded and standardized structures. Expansion is off by default and does not guess what R groups mean.
Geometry checks are off by default. Use `--geometry-eligibility` to reject molecules with net charge, radicals, or unsupported elements before embedding or labeling. The calculator must list its supported elements, or you must supply them with `--allowed-elements H C N O ...`. Checks include implicit hydrogens. Use `--require-assigned-stereo` to also reject unassigned tetrahedral or double-bond stereo. Either extra flag turns on the checks. Checks run after standardization. Neutral groups with separate charges, such as nitro, are allowed. The manifest lists all reasons a molecule was rejected.
To create train, validation, and test files, run [`scripts/split_forcefield.py`](scripts/split_forcefield.py) `--frames data.extxyz --manifest data.manifest.jsonl --out-dir splits --seed 0`. The same seed gives the same split for each molecule, regardless of input order. All conformers, equivalent SMILES, and stereoisomers stay together. The default fractions are 0.8/0.1/0.1. Use `--fractions` to change them. Small datasets may have empty splits. Molecules with the same connectivity stay in one split. Related molecular scaffolds may still appear in different splits. Output manifests keep the frame matches and indices. Skipped inputs have a separate manifest. Missing or duplicate frame matches cause an error. The output directory must be empty.
[`scripts/image_to_forcefield.py`](scripts/image_to_forcefield.py) `--displaced-samples 3 --max-displacement 0.1` tries three random coordinate changes for each ETKDG conformer, then computes energies and forces. The seed makes the changes repeatable. Each atom moves at most 0.1 Angstrom. Samples are rejected if atoms are closer than `--displacement-min-distance` (default 0.6 Angstrom) or if specified stereochemistry changes. Rejected samples are not retried. The manifest records all attempts and sampling settings. Frames and conformers include parent and sample IDs. Sample zero is the original. Sampling is off by default and cannot be used with relaxation. Whether these extra samples improve force-field training has not been tested.

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
