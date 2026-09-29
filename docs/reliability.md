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
