# Interactive rotation lab

Open `notebooks/05_rotation_demo.ipynb` to rotate a synthetic molecule with a slider,
inspect feature equivariance, and optionally compare predictions from a trained checkpoint.
The same interface can run as a local browser app. Everything runs on your Python host;
there are no external UI assets, datasets, hosted APIs, or automatically downloaded weights.

## How to run

Use Python 3.11 (3.13 is incompatible with the escnn dependency stack). From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
python -m jupyterlab notebooks/05_rotation_demo.ipynb
```

`py3nj`, an escnn dependency, may need a Fortran compiler: `brew install gcc` on macOS,
or `sudo apt-get install gfortran` on Debian/Ubuntu. The demo requirements avoid the full
training environment's CUDA and MACE dependencies. An existing project environment works too;
install JupyterLab there, or point the notebook's `MODEL_PYTHON` at that environment's Python.
The notebook kernel and model interpreter can be different.

To open just the browser UI, without Jupyter (using an environment with model dependencies):

```bash
python -m ouroboros.demo
# Open the session URL printed in the terminal. Stop with Ctrl+C.
```

The server listens only on `127.0.0.1`. Use `--port 0` to choose an available port.
An unguessable session path scopes requests; this is a local research tool, not a public server.
Each server has one active model shared by its browser tabs.

For Colab, run `00_colab_setup.ipynb` in the same runtime first, with a checkout containing the
demo. Then run `05_rotation_demo.ipynb`. It launches calculations using `/content/venv/bin/python`
and embeds the UI using Colab's
[official iframe helper](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py).
The default example needs only CPU.
The existing setup notebook's branch selection must point to the demo branch until it is merged.
For other remote Jupyter servers, forward the printed local port with SSH; the notebook iframe
assumes a local browser unless it detects Colab.

## What the controls do

- **Example / Reference SMILES / Draw molecule:** generate a known structure with the repository's
  renderer. The ground-truth label is re-derived from its stereochemical drawing. Rendering keeps
  the molecule inside an inscribed circle so rotation does not clip it.
- **Angle slider:** preview rotation while dragging; calculate after release. The backend uses
  the same `rotate_images` as evaluation: exact pixel permutations at 90° multiples, bilinear
  interpolation otherwise. Shortcuts provide 0°, 45°, 90°, 180° and 270°.
- **Random demo network / Use random demo:** select C4, C8 or C16. These small seeded networks
  demonstrate architectural symmetry, never trained recognition. Recognition is disabled.
- **Load checkpoint:** read the configuration, vocabulary and weights from a trusted training
  checkpoint. Choose CPU or CUDA. No training datasets or external vocabulary files are needed.

The original image's features and prediction are cached while rotating the same molecule.
Changing the molecule or model invalidates that cache. An invalid load preserves the old session.

## What the numbers mean

Feature error is `max(abs(f(Rx) - rho(R)f(x))) / max(abs(rho(R)f(x)))`, with a denominator floor
of `1e-12`. `rho(R)` includes spatial rotation **and the orientation-channel permutation**.
This is evaluated only when the selected angle belongs to the encoder's C_N group. For other
angles it says **Not defined**, not zero. Non-90° group angles introduce interpolation error.
Heatmaps average channel magnitudes for display and share one scale; the scalar metric uses
all channels. A/B/E encoders have no feature group action exposed by this demo.

Decoder first-step change uses the same relative maximum error on logits conditioned only on
the BOS token. It is a rotation-sensitivity probe for every angle and encoder, including baseline
checkpoints. It is not recognition accuracy and does not prove invariance of later tokens.
Calculations use float32 evaluation mode without mixed precision.

With a checkpoint, the UI independently decodes each orientation with the model's full configured
greedy length. It shows raw predicted SMILES and redraws, validity, stereo-aware exact match,
connectivity match and the number of correctly recovered stereo elements. It uses `score_pair`
and the same standardization as evaluation (including neutral largest-fragment selection).
Prediction-to-prediction agreement compares full canonical isomeric SMILES; invalid predictions
have no identity-agreement result. Two matching predictions can still both be wrong.

This synthetic single-molecule demonstration is not a real-document evaluation. No trained OCSR
weights are bundled. Scoring parity with the motivating paper remains unverified, as in the
repository's evaluation reports. The downstream RDKit conformer/MACE stage is outside this demo.

## Development

```bash
python scripts/make_rotation_demo_nb.py  # regenerate this notebook only
python -m pytest -q tests/test_rotation_demo.py tests/test_equivariance.py
```

Notebook outputs are intentionally cleared so session URLs, local paths and predictions from
private checkpoints are not committed. The HTML interface ships as package data.
