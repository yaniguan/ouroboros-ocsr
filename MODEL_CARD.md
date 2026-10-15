# Ouroboros v0.1.0-alpha model card

## Intended use

Research and education: compare random rotation augmentation and a discrete rotation prior
on synthetic molecular drawings. The local UI renders reference SMILES; it is not a general
document extraction product. No real-document accuracy, production reliability or external
paper scoring parity is established. Verify predictions, particularly stereochemistry.

## Architecture and training

| Arm | Encoder parameters | Decoder parameters | Total |
| --- | ---: | ---: | ---: |
| A | 17,812,416 | 25,358,848 | 43,171,264 |
| B | 17,812,416 | 25,358,848 | 43,171,264 |
| C | 7,024,300 | 25,358,848 | 32,383,148 |

A/B: ResNet-18-style CNN, 144 image tokens at 384 px, absolute image positions and a two-layer
Transformer mixer. C: C8 steerable CNN, fields [7,14,27,55], group pooling and relative-position
mixer. Shared decoder: six layers, dimension 512, eight heads, feed-forward dimension 2048,
maximum length 128. Teacher forcing in training; greedy decoding in inference. No pretraining.

Seed 0; 200,000 synthetic examples; batch 32; maximum 10,000 steps; AdamW; learning rate
0.0003; 1,000 warmup steps; bf16. B alone uses uniform random rotations. All arms use the same
fixed 1,000 validation examples, checked every 1,000 steps, patience 5, minimum progress 0.001.
The best committed validation checkpoint is selected before test. Test results do not tune it.

## Data and provenance

MOSES/ZINC-derived molecules, rendered with RDKit. Stereo is synthetically assigned before
rendering, not preserved from source databases. The corpus specifies 40% stereo examples.
The pilot selects 200,000 usable examples from 201 train shards. Four records were already
missing after rendering. Three additional assigned/stored stereo mismatches were quarantined:
`train_00011370`, `train_00057670`, `train_00167450`. Originals are retained, the train prefix
is replenished, and validation/test remain unchanged.

Strict audits check labels, image/metadata pairing, dimensions and zero cross-split molecular
connectivity overlap. This is not the full 1M audit. Receipts and sample identities are under
`benchmarks/v0.1.0-alpha/`. Inference exports contain tensors, vocabulary, model configuration,
step and source checkpoint hash; no optimizer, RNG, dataset or private training paths.

## Evaluation and limitations

See [measured results](docs/v0-results.md). Full test: 10,000 rendered examples at 0°.
Rotation: the same 2,000-example prefix at 24 angles, 0–345° in 15° steps. The rotation mean
is an equal-weight angle average, not 48,000 independent molecules. Bootstrap intervals
are per-set uncertainty; they do not measure training-seed variability.

Primary exact match includes stereo after repository RDKit standardization. Connectivity-only
accuracy and stereo-subset accuracy are separate. External-paper scoring parity is unverified.
Known limits: one seed, synthetic rendering, no human-verified real holdout, unmatched parameter
counts, hardware-dependent speed, stereo errors, no beam search or calibrated confidence.
Consistency alone is not correctness. No state-of-the-art or real-document claim is made.

## Distribution

Project code and project-produced inference weights are distributed under the included MIT
license. Source datasets and dependencies retain their own terms; see [DATA.md](DATA.md).
No MACE weights or third-party dataset archives are bundled. Official exports support
`torch.load(..., weights_only=True)` and the downloader verifies SHA256. The demo also accepts
trusted legacy training checkpoints; only load files from sources you trust.
