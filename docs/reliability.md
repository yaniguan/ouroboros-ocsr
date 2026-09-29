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
