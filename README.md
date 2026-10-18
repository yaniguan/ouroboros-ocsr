# Ouroboros — rotate the drawing, compare the recognition

Ouroboros is a research preview of rotation-aware optical chemical structure recognition
(OCSR): molecule drawing → SMILES with stereochemistry. The local rotation lab compares
predictions before and after rotating a synthetic drawing.

**v0.1.0-alpha:** trained A/B/C checkpoints, a shared evaluation protocol, and a local demo.
This is a synthetic-data, single-seed pilot. Real-document accuracy has not been established.
See [results](docs/v0-results.md), [model card](MODEL_CARD.md), and
[release acceptance](docs/v0-release.md). The original hypotheses remain in the
[research plan](docs/research-plan.md); they are not rewritten to fit observations.

| Arm | Image encoder | Random rotation in training |
| --- | --- | --- |
| A | ResNet-style CNN + absolute positions + Transformer mixer | No |
| B | Same as A | Yes, uniform 0–360° |
| C | C8 steerable CNN + invariant token-set mixer | No |

All arms use the same six-layer decoder, 200,000 training examples, batch 32,
10,000 maximum steps, and the same 1,000 validation examples. C uses the FLOP-matched
configuration; parameter counts are **not** matched.

## How to run

Use **Python 3.11** and C/Fortran compilers. On macOS install `gcc` with Homebrew;
on Ubuntu install `gfortran` and `build-essential`. No GPU or dataset is needed for the demo.

```bash
git clone --branch v0.1.0-alpha https://github.com/yaniguan/ouroboros-ocsr.git
cd ouroboros-ocsr
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python scripts/prepare_inference.py
python -m pip install --no-deps -e .
python -m ouroboros.weights fetch --arm all
python -m ouroboros.demo --checkpoint checkpoints/ouroboros-C.pt
```

Open the session URL printed in the terminal. Draw a molecule, move the angle slider,
and compare predictions with the reference. To switch models, enter the absolute path to
`checkpoints/ouroboros-A.pt`, `ouroboros-B.pt`, or `ouroboros-C.pt` in **Load checkpoint**.
CUDA users can add `--device cuda` to the launch command.

The downloader checks release SHA256 hashes. `prepare_inference.py` restores two upstream
data tables omitted from macOS source-built `lie-learn` wheels, using pinned, verified files.
For a notebook interface, see the [rotation lab guide](docs/rotation-demo.md).

For an architecture-only demonstration without downloading weights:

```bash
python -m ouroboros.demo
```

That mode uses a random network and disables recognition. A loaded model can produce
matching but incorrect predictions; the UI distinguishes correctness from consistency.

## Reproduce and extend

Summaries and per-angle metrics are in `benchmarks/v0.1.0-alpha/`. Regenerate the report:

```bash
python -m pip install matplotlib
python scripts/report_v0.py
```

For training, use the Linux CUDA pins in `requirements.txt`,
`configs/pilot-{A,B,C}-200k.yaml`, and the [reproduction protocol](docs/v0-release.md).
Training requires the synthetic corpus, which is not bundled. Selected-data hashes and
three quarantined train records are documented.

## Scope

- No established accuracy on journal pages, patents, handwriting or arbitrary uploaded images.
  The demo renders examples from reference SMILES.
- C8 is a discrete rotation prior. Rasterization, interpolation and numerical precision mean
  predictions need not be identical at every angle.
- Stereochemistry remains an error source. A valid SMILES is not proof of correct recognition.
- The older baseline ≥70% target is not marked complete by this preview release.
- Optional 3D/MACE research tools are outside the demo acceptance criteria; see [DATA.md](DATA.md).

Code is MIT-licensed. Weights and dependency provenance: [MODEL_CARD.md](MODEL_CARD.md).
Experiment history: [TASK.md](TASK.md).
