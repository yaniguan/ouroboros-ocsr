# v0.1.0-alpha acceptance and reproduction

A research preview is done when a new user can run the models, compare A/B/C under the published
protocol, regenerate the results, and understand limitations. A positive equivariance result
is not a release requirement; the original baseline ≥70% milestone remains separate.

## Release gates

- [ ] A/B/C training and full-test/24-angle evaluation completed successfully.
- [ ] Best-checkpoint selection, identical data selections and A/B configuration parity verified.
- [ ] All three inference exports preserve model tensors and load in the demo.
- [ ] Clean environment installation and demo request/response flow verified.
- [ ] Regression tests, lint and wheel build pass.
- [ ] Results, hardware differences, single-seed scope and stereo limitations documented.
- [ ] Versioned source, wheel, weights and SHA256 receipts available in the GitHub release.
- [ ] Published download and installation path verified.

## How to run

Follow the [README](../README.md#how-to-run) for inference. Python 3.11 is tested.
Use the exact Linux CUDA requirements for training. macOS inference uses the lightweight
requirements and verified upstream `lie-learn` data tables.

## Training reproduction

The synthetic corpus is described in DATA.md. Match hashes and sample identities in
`benchmarks/v0.1.0-alpha/data-transfer-receipts.json` and `data-audit.json`. Quarantine the
three documented train pairs and manifest rows into a derived view, retain originals, and
leave validation/test unchanged. Run the strict audit before training.

The corpus is not bundled. Inference and report regeneration do not need it; without the
original corpus, a byte-identical training rerun cannot be claimed.

With audited shards at `/absolute/pilot/shards`, run each arm in a new directory:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m ouroboros.train --config configs/pilot-B-200k.yaml --out runs/B_n200k_f0_s0 --set data.synthetic_root=/absolute/pilot/shards validation.root=/absolute/pilot/shards eval.sets.rendered_test.root=/absolute/pilot/shards data.num_workers=4 train.device=cuda eval.batch_size=32
python scripts/evaluate.py --run runs/B_n200k_f0_s0
```

Repeat with A and C configurations and distinct output directories. Native bf16 CUDA hardware
is required. The HPC helpers retain fail-closed data, capacity and resume checks. Inference
exports cannot resume optimizer state. Evaluate the validation-selected checkpoint.

## Next release

Multiple seeds, human-verified real-document holdouts, parameter-matched ablations, larger
sweeps and downstream 3D/MACE error propagation. Pre-registered hypotheses remain unchanged;
measured outcomes are reported separately.
