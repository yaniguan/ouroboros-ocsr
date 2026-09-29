"""Generate only notebooks/05_rotation_demo.ipynb; leave the experiment notebooks unchanged."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def cell(kind, source):
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook():
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell(
                "markdown",
                """
                # Ouroboros — interactive rotation lab

                **Drag a slider to rotate a molecule and measure the network's response.**
                The default is a small random C4 model on CPU: no dataset, training, GPU or
                checkpoint is required. Recognition stays off until you load your own weights.

                **How to run locally** (from the repository root, Python 3.11 recommended):
                ```bash
                python3.11 -m venv .venv
                source .venv/bin/activate
                pip install -r requirements-demo.txt
                python -m jupyterlab notebooks/05_rotation_demo.ipynb
                ```
                If `py3nj` needs to compile, install `gfortran` first (`brew install gcc` on
                macOS; `sudo apt-get install gfortran` on Debian/Ubuntu). You can also use your
                existing project environment. See [the guide](../docs/rotation-demo.md).

                **Colab:** first run `00_colab_setup.ipynb` in the same runtime, then open this
                notebook. Ensure that checkout contains this demo (merged `main`, or the demo
                branch). The UI uses Colab's kernel, while calculations run in `/content/venv`;
                do not install escnn into Colab's system Python. No GPU is needed by default.

                Run the cells below in order, then use the embedded UI or its full-page link.
            """,
            ),
            cell(
                "code",
                """
                import os
                import sys
                from pathlib import Path

                # Set these paths explicitly if your notebook is outside the repository.
                REPO_ROOT = next(
                    (p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / "ouroboros" / "demo" / "index.html").is_file()),
                    Path("/content/ouroboros-ocsr"),
                )
                candidates = [Path("/content/venv/bin/python"), REPO_ROOT / ".venv/bin/python"]
                MODEL_PYTHON = os.environ.get(
                    "OUROBOROS_DEMO_PYTHON",
                    str(next((p for p in candidates if p.is_file()), Path(sys.executable))),
                )
                PORT = 0  # choose an available local port
                CHECKPOINT = ""  # optional: /path/to/runs/my_run/ckpt/last.pt
                DEVICE = "cpu"  # set to "cuda" only for a trained model on a CUDA host
                print("Repository:", REPO_ROOT)
                print("Model interpreter:", MODEL_PYTHON)
            """,
            ),
            cell(
                "code",
                """
                import json
                import subprocess
                import tempfile
                import time
                import urllib.request

                # Rerunning this cell stops only the server previously started by this notebook.
                if "DEMO_PROCESS" in globals() and DEMO_PROCESS.poll() is None:
                    DEMO_PROCESS.terminate()
                    DEMO_PROCESS.wait(timeout=15)
                if not (REPO_ROOT / "ouroboros/demo/index.html").is_file():
                    raise RuntimeError("Set REPO_ROOT to the checkout containing ouroboros/demo.")
                DEMO_DIR = Path(tempfile.mkdtemp(prefix="ouroboros-rotation-"))
                ready_file = DEMO_DIR / "ready.json"
                log_file = DEMO_DIR / "server.log"
                command = [MODEL_PYTHON, "-m", "ouroboros.demo", "--port", str(PORT),
                           "--ready-file", str(ready_file)]
                if CHECKPOINT:
                    command += ["--checkpoint", CHECKPOINT, "--device", DEVICE]
                env = dict(os.environ)
                env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                env["MPLBACKEND"] = "Agg"
                with log_file.open("w") as log:
                    DEMO_PROCESS = subprocess.Popen(
                        command, cwd=REPO_ROOT, env=env, stdout=log, stderr=log,
                    )
                deadline = time.monotonic() + 120
                while not ready_file.exists():
                    if DEMO_PROCESS.poll() is not None:
                        message = "Demo startup failed. Check MODEL_PYTHON and dependencies.\\n"
                        raise RuntimeError(message + log_file.read_text())
                    if time.monotonic() > deadline:
                        DEMO_PROCESS.terminate()
                        DEMO_PROCESS.wait(timeout=15)
                        raise TimeoutError(f"Demo startup timed out; inspect {log_file}")
                    time.sleep(0.2)
                DEMO_INFO = json.loads(ready_file.read_text())
                with urllib.request.urlopen(DEMO_INFO["url"] + "health", timeout=10) as response:
                    assert json.load(response)["ready"]
                print("Rotation lab ready:", DEMO_INFO["url"])
                print("Server log:", log_file)
            """,
            ),
            cell(
                "code",
                """
                from html import escape
                from IPython.display import HTML, IFrame, display

                demo_url = DEMO_INFO["url"]
                try:
                    from google.colab import output
                except ImportError:
                    # Local Jupyter: browser and model server are on the same machine.
                    display(HTML(f'<p><a href="{escape(demo_url, quote=True)}" target="_blank" '
                                 'rel="noopener">Open the full-page rotation lab ↗</a></p>'))
                    display(IFrame(demo_url, width="100%", height=1450))
                else:
                    output.serve_kernel_port_as_iframe(
                        DEMO_INFO["port"], path=DEMO_INFO["path"], height=1450,
                    )
            """,
            ),
            cell(
                "markdown",
                """
                ## What to try

                1. Compare **0°, 90°, 180°, 270°** in random C4 mode. Feature errors should
                   be close to floating-point precision even though no training has happened.
                2. Drag to **30°**. The feature error becomes **Not defined**, because 30° is
                   not in C4. Decoder first-step change still measures sensitivity at this angle.
                3. Select **C8**, press **Use random demo**, then try **45°**. It belongs to C8,
                   but interpolation on pixels introduces an approximate-equivariance error.
                4. Open **See the feature comparison**. The expected transformation rotates
                   spatial features and permutes their orientation channels. Heatmaps share a
                   scale; the scalar metric compares all channels before visualization.
                5. Paste a checkpoint path on the Python host and click **Load checkpoint**.
                   Compare original/rotated greedy SMILES, molecule redraws and ground-truth
                   scores. The ground truth is used for scoring only, never for decoding.

                **Interpretation:** symmetry is not accuracy. Two orientations can agree on a
                wrong answer. The first-step logit probe is not a guarantee about the full
                decoded sequence. Ground-truth matching uses the repository's stereo-aware
                standardization; this is a single synthetic example, not a benchmark result.

                The UI preserves the previous working result if a SMILES or checkpoint is invalid.
                A/B/E checkpoints can be used for recognition, but do not expose a C_N feature
                action. C/C+/D checkpoints also support feature-equivariance measurement.

                Your checkpoint already contains its configuration and vocabulary. Moving it
                from Drive to another machine does not require the old dataset or vocab paths.
                Only load your own trusted PyTorch checkpoints. The notebook does not train,
                download weights, or run the 3D/MACE stage.
            """,
            ),
            cell(
                "code",
                """
                # Optional cleanup, after you finish using the UI:
                # DEMO_PROCESS.terminate()
                # DEMO_PROCESS.wait(timeout=15)
            """,
            ),
        ],
    }


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "notebooks/05_rotation_demo.ipynb"
    path.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")
