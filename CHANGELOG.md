# Changelog

## 2026-11 — Holdout evaluation, lab modules, 3D workbench & codebase audit

- Froze the 300-image `frozen-test-v2` holdout (`ACS` 100, `UOB` 100, `USPTO` 100) and scored it once with `PMC val`-locked decoding (`AC-ensemble-beam5-ttapad`): `46.7%` exact match overall (`85.0%` `UOB`, `30.0%` `ACS`, `25.0%` `USPTO`; `benchmarks/frozen-test-v2/`).
- Added the 200-image `DECIMER Hand-drawn` evaluation subset (`4.5%` exact match for fine-tune v2 A + C beam 5; `benchmarks/handdrawn-decimer/`).
- Locked selective confidence-gate limits (`min_margin=0.05`, `max_disagreement=0.0`) on `PMC val` (`92.19%` precision at `19.9%` coverage) and scored `frozen-test-v2` once (`97.78%` exact precision across `88/90` accepted figures, `30.0%` coverage; `benchmarks/selective-gate/`).
- Added the `ouroboros.infer` CPU `int8` inference API (`benchmarks/cpu-api/`), `ouroboros.labvision` 96-well rectangular & circular plate finder, `lab.molecule_context` (`ouroboros.molecule_context/v1`), `lab.edge_plan`, `lab.quantize` (`benchmarks/quantize-cpu/`), and `lab.grpo_verifier`.
- Launched the interactive project website and zero-install Browser 3D Lab (`site/`, `ouroboros.ui`), enforced `ruff check .` in CI, and hardened cross-platform Windows/CUDA/chemistry/API edge cases.

## 2026-10-30 — Real-figure fine-tune v2

- Fine-tuned arms A and C on real PMC figures (`colab/finetune_v2.ipynb`), selecting checkpoints on the molecule-disjoint real validation split.
- On the 298-image real benchmark sample (ACS 100, UOB 100, USPTO 98), the A + C ensemble reached 54.0% exact match with stereochemistry (beam 5 + TTA) and 52.3% without TTA; single-model greedy scores were 51.0% for A and 37.6% for C (`benchmarks/finetune-v2-oct30/`).

## 2026-10-15 — Baseline B and v0.1.0-alpha weights

- Completed rotation-augmented baseline B (seed 0, 200,000 synthetic drawings, 10,000 bf16 steps) on a separate A6000 GPU cluster: 64.41% test exact at 0° and 65.7% mean across 24 angles (`docs/v0-results.md`).
- Exported and SHA256-verified A/B/C inference checkpoints for the `v0.1.0-alpha` GitHub prerelease (`benchmarks/v0.1.0-alpha/`).

## v0.1.0-alpha — 2026-10-01

- Trained and evaluated synthetic pilots for arm A (68.69% exact at 0°, 25.4% 24-angle mean) and C8 steerable arm C (67.08% exact at 0°, 65.8% 24-angle mean) on A100-40GB (`docs/v0-results.md`).
- Verified MACE-OFF23 mirror-conformer energy parity across 20 enantiomer pairs (`max |dE| = 6.48e-10 eV`, `benchmarks/hpc-20260930/enantiomers.json`).
- A/B/C research-preview protocol with shared validation, best-checkpoint selection and full-test/rotation reporting.
- Inference checkpoint exports and checksum-verified release downloads.
- Local rotation lab and notebook: predictions, redraws, stereo-aware scores and diagnostics.
- Model card, reproduction guide and explicit synthetic-only, single-seed limitations.
- Native bf16 checks reject emulation before training; validation/RNG/best-checkpoint state survives interrupted runs.
- macOS inference installation restores missing upstream lie-learn tables with verified hashes.

Optional geometry/MACE research tools remain outside the demo acceptance scope.
