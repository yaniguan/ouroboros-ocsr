---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-to-text
tags:
- chemistry
- optical-chemical-structure-recognition
- ocsr
- smiles
- equivariance
- steerable-cnn
datasets:
- yaniguan/ocsr-dataset
---
<!--
Template for scripts/export_hf_release.py. Every double-brace placeholder is filled from the checkpoint,
config.json, metrics.json (--report / --profiling) or the command line; the exporter refuses to
write a card with an unfilled placeholder. Edit the prose here, not in the generated README.md.
-->

# {{model_name}}

Optical chemical structure recognition (OCSR): a molecule drawing in, a SMILES string out.
Arm **{{arm}}** of [Ouroboros](https://github.com/yaniguan/ouroboros-ocsr): a
**{{encoder_type}}** image encoder ({{symmetry_group}}) and a Transformer SMILES decoder,
fine-tuned on real document images. Exported from training step {{step}}
(source checkpoint SHA256 `{{source_checkpoint_sha256}}`).

| | |
| --- | --- |
| Input | one grayscale image, {{image_size}} × {{image_size}}, ink = 1 − gray/255, white = 0 |
| Output | SMILES (RDKit-style, stereo tokens `@`, `@@`, `/`, `\`; R-group pseudo-atoms `*`, `[R1]`, …) |
| Parameters | {{n_params_total}} total, {{n_params_encoder}} in the encoder |
| Vocabulary | {{vocab_size}} tokens (`vocab.json`, append-only over the released vocabulary) |
| Weights | `{{weights_file}}` (full model), `{{encoder_weights_file}}` (encoder only, `encoder.` prefix stripped) |

## License

**The weights are released under CC-BY-NC-4.0** (non-commercial). They were fine-tuned on
VERDICT-PMC (`yaniguan/ocsr-dataset`), whose figure crops come from PubMed Central articles
distributed under CC-BY-NC terms; the non-commercial restriction is carried over to the weights.
The Ouroboros source code remains MIT-licensed. MACE-OFF weights (used only by the optional 3D
stage, Academic Software License) are not part of this repository.

## Intended use

- Research on OCSR and on rotation priors for document images; benchmarking; building candidate
  structure lists that a chemist verifies.
- Not intended for unattended extraction into databases, regulatory, clinical or safety
  decisions, or any use where an unverified structure (especially its stereochemistry) causes
  harm. There is no calibrated confidence: a fluent, valid SMILES can still be wrong.

## Symmetry: rotation-only C_N equivariance

The steerable encoder is built with `escnn` on `rot2dOnR2(N)`: its feature maps transform
predictably under the N planar rotations of C_N, and group pooling plus a token mixer that only
sees pairwise token *distances* make the token set rotation-invariant. Rotations by multiples of
90° are exact on the pixel grid; the intermediate C8/C16 rotations are approximate (interpolation),
which is measured in the repository rather than assumed.

**Reflections are deliberately excluded** (no `flipRot2dOnR2`, no D_N). Mirroring a drawing turns
every wedge/hash stereocentre into its enantiomer: a reflection-invariant encoder could not tell
(R) from (S). Equivariance is a property of the image encoder only; it is an inductive bias, not a
guarantee of accuracy on rotated pages, and it says nothing about E(3) symmetry of the molecule.

## Training data

1. **Pre-training** (released v0.1.0-alpha weights): 200,000 RDKit renderings of MOSES/ZINC
   molecules, 40% with synthetically assigned stereo; see `MODEL_CARD.md` in the repository.
2. **Fine-tuning**: VERDICT-PMC real figure crops (`yaniguan/ocsr-dataset`, train split, pinned
   commit) mixed with real-style synthetic renderings (ZINC250k, R-group substitution) at a fixed
   real fraction per batch. Checkpoint selection uses exact match on the PMC validation split,
   which is molecule-disjoint from PMC train; the step-0 (released) weights are scored too, so
   selection can fall back to them.
3. **Leakage gate**: every PMC (train and val) and synthetic training row whose InChIKey
   *connectivity block* matches a held-out test molecule is removed (so stereoisomers of test
   molecules are excluded as well), and `scripts/check_leakage.py --fail-on-connectivity` must
   report zero overlap before training starts.

Run configuration recorded in the checkpoint:

{{training_summary}}

## Evaluation

Held-out real test set: the seeded `Keylab/OCSR-Benchmarks` subset (100 images each from ACS, UOB
and USPTO; 298 scoreable after label parsing), never used for training or selection.
Exact match = canonical SMILES equality after the repository's RDKit standardization, **full
stereochemistry included**; 95% CI = percentile bootstrap over images. Greedy unless stated.

{{eval_table}}

{{eval_notes}}

### Inference cost

{{profiling_table}}

## Limitations

- **R-groups and abbreviations**: labels such as `R1`, `Ar`, `OMe`-style superatoms are
  decoded into a small fixed set of pseudo-atom tokens; unusual labels, variable attachment
  points and Markush notation are frequently wrong. ACS figures are the hardest subset.
- **Long molecules**: accuracy falls sharply beyond ~60 SMILES tokens; the decoder cannot
  emit more than its maximum length at all.
- **Stereochemistry**: wedge/hash interpretation is the most common error among otherwise
  correct constitutions; always check stereocentres and double-bond geometry.
- Small test set (298 images): differences of a few points are within the confidence intervals.
- Scoring parity with external papers is not verified; numbers are not directly comparable
  with published tables unless the same standardization is used.
- Single training seed; no calibrated confidence; one molecule per image.

## How to use

```python
from huggingface_hub import snapshot_download
from ouroboros.hf import load_pretrained

model = load_pretrained(snapshot_download("{{repo_id}}"))  # eval mode, CPU
smiles = model.generate(images)  # images: float tensor [B, 1, {{image_size}}, {{image_size}}]
```

Use `ouroboros.data.real.fit_to_canvas` to letterbox a document crop onto the square canvas
before converting it to ink. `checksums.txt` lists the SHA256 of every file in this folder.

## Citation

```bibtex
@software{ouroboros_ocsr,
  author = {{{author}}},
  title  = {Ouroboros: rotation-equivariant optical chemical structure recognition},
  url    = {https://github.com/yaniguan/ouroboros-ocsr},
  year   = {2026}
}
```

Please also cite the VERDICT companion paper (arXiv:2608.22183) for the VERDICT-PMC data, and
`escnn` (Cesa, Lang & Weiler, ICLR 2022) for the steerable layers.

_Card generated on {{date}} by `scripts/export_hf_release.py`._
