# Reliable runs and migration

Use one writer per run, rendering directory, evaluation directory or cache. Temporary files
are published by same-directory rename; metadata writes and checkpoints are flushed first.

## Training resume

New checkpoints bind the full configuration (including seed, batch size and scheduler horizon),
ordered vocabulary, actual training shard bytes and selected sample order. Synthetic and real
training datasets are checked separately. A mismatch raises before model state or existing
configuration/log files are changed. Worker-count/device changes are configuration changes too.
Keep inputs immutable during a run. Hashing happens on startup, not every optimizer step.

Old checkpoints without identity metadata can still be evaluated, but cannot safely resume
training: use a new run directory. Nothing is inferred about their original dataset bytes.
The fixed-step training protocol, mixture sampling and model architecture remain unchanged.

## Geometry

Embedding has a 30-second RDKit timeout **per conformer/fragment**, with a random-coordinate retry.
Negative or missing conformer IDs are rejected. Only finite, converged, stereo-preserving relaxed
candidates can supply the reported energy. Failure reasons and embedding attempt diagnostics are
retained. Mirrored enantiomer geometry is still evaluated explicitly; energy differences remain
restricted to equal molecular formulae. Invalid geometry contributes no energy-error observation.

## Generated data

Composition reuse checks the pool, complete exclusion set and stereo-assignment implementation.
Rendering binds the ordered manifest, image size, seed, shard size, renderer code and relevant
library versions in `<split>.generation.json`. Each completed tar has a SHA256 receipt. Resume
verifies existing bytes and rejects changed inputs, missing receipts or unexpected shards.
Interrupted `.part` files are ignored. Legacy render directories need regeneration into a new
output directory. Extending a dry-run/prefix also needs a new output directory.
The PNG/JSON shard format and connectivity-based split policy are unchanged. Loader index caches
are bound to tar contents and rebuild after content changes or incomplete metadata writes.

## Evaluation and energy caches

`scripts/evaluate.py` binds checkpoint bytes, vocabulary, config, data/selection, angles, batch
size, device, library versions and implementation to `eval/{full,sweep}/evaluation.json`.
Completed batches are atomic JSON cache entries with checksums. Restart uses those entries;
changing any identity input requires a new evaluation output (use a new run directory).
The callable `run_eval` API only reuses results when an explicit identity is supplied.

`scripts/error_propagation.py --cache DIR` persists each search and mirrored-geometry job
(default: `<out>/energy-cache`). Keys include actual MACE model weights, conformer settings,
seed, versions and implementation. Failed geometry is retained too. To retry a failed job,
use a new cache directory. Pair/summary files are regenerated from cached molecular jobs;
rendered/real separation, bootstrap CIs and formula/mirror controls are unchanged.

## Source downloads and filtering

ZINC250k and MOSES now use immutable upstream revisions. ZINC bytes are checked against the
Git blob identity and size; MOSES against the upstream Git LFS SHA256 and size. A local SHA256
receipt is written only after verification. Existing source files are verified too. Downloads
retain a source-bound `.part` file, request the remaining HTTP range on restart, and restart
from byte zero when the server ignores Range. Wrong ranges and checksums never publish a source.

Filtering stores the cursor, rejection counts and accepted connectivity keys transactionally in
`<pool>.filter.sqlite`. Restart scans/spools the input to verify its identity, but skips chemistry
work for committed batches. Input order/content, RDKit and filtering code changes are rejected.
Deduplication, stereo assignment and split rules stay unchanged; ChEMBL is not made the default.
An explicitly supplied existing `--pool` is still reusable and its bytes bind composition.
Legacy default pools without filtering metadata require a new output path. The SQLite cache is
generated data and belongs beside the pool outside version control.

## How to run the GPU probes

Use the installed project environment and generated experiment configs. Each output must be new.
For an A100 or another bf16-capable CUDA GPU:

```bash
# Repeat with the A/C/D config you actually intend to train.
python scripts/benchmark_cuda.py --config configs/sweep/D_n200k_f0_s0.yaml \
  --output runs/probes/D-capacity.json --batch 32 --steps 20 --warmup 3
# For a 512-token probe, explicitly increase the decoder capacity in this probe's config.
python scripts/benchmark_cuda.py --config configs/sweep/D_n200k_f0_s0.yaml \
  --output runs/probes/D-512-capacity.json --batch 32 --sequence-length 512 \
  --set model.decoder.max_len=512
# Existing dataset shards and matching vocabulary are required for the resume comparison.
python scripts/benchmark_cuda_resume.py --config configs/sweep/A_n200k_f0_s0.yaml \
  --output runs/probes/A-resume --checkpoint-step 25 --compared-steps 100 \
  --set data.synthetic_root=/content/data/full/shards data.synthetic_max_samples=2000
```

Capacity uses the current model, vocabulary, AdamW parameter groups, learning-rate scheduler,
augmentation and loss. Inputs are random tensors; data I/O and validation are excluded. Reports
include source/config identities, versions, synchronized optimizer timings, CUDA peak allocated/
reserved bytes and a stated warmup exclusion. Capacity results are not accuracy or full-run costs.

The resume probe uses the actual Trainer and configured synthetic/real mixture, interrupts after
a checkpoint, then compares the next 100 losses with uninterrupted training. It fails if the mean
relative error exceeds 1% (configurable). It writes checkpoints/logs under `runs/`, never under
`benchmarks/`. Small CPU smoke checks use `--device cpu --precision fp32` and suitably reduced
model/data config; their reports explicitly carry `cpu_smoke_only: true` and no GPU memory values.
Historical A100 reports in `benchmarks/legacy-local/` belong to the other implementation and must
not be substituted for measurements of the current model.

## Local verification (2026-09-28)

The full suite passed **162 tests** on macOS/Python 3.11 using the existing local environment
(torch 2.8 / RDKit 2025.9) plus temporary PyArrow/Hugging Face test dependencies. `ruff check .`
passed. The final rotation-decoding and benchmark changes also passed their 8 targeted tests.
This is a compatibility check, not validation of the repository's pinned CUDA environment.
No CUDA/A100 execution or current-model GPU performance claim was made during this migration.
