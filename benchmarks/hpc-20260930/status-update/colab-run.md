# Colab A/C pilot — 2026-09-30

Notebook: https://colab.research.google.com/drive/1I5HwAk544O9GeOccSDTyMx4A5izYNz2P?authuser=1

Run ID: `colab-20260930-ac-s0`. Initial pipeline PID: `4243`.
Hardware: NVIDIA A100-SXM4-40GB, 40960 MiB. Existing Colab Pro quota only.

The new cells immediately after the initial hardware probe mount Drive, upload
and hash-check the current source, launch the pipeline and report its status.
Older cells below them belong to the legacy implementation; execute the new
cells individually. The source archive includes the shared-validation patch.

Output location: `MyDrive/ouroboros/colab-20260930-ac-s0/`.

- `source.zip`, `source.json`: exact source snapshot and per-file hashes.
- `reports/pipeline-status.json`: latest stage and exit status.
- `pipeline.log`, `logs/*.log`: orchestration and stage logs.
- `reports/`: tests, data audit, capacity and resume evidence.
- `runs/A_n200k_f0_s0`, `runs/C_n200k_f0_s0`: configs, training logs and checkpoints.

The pipeline verifies the pinned environment, native bf16, every transferred
file hash, full tests, all 201 training + 5 validation + 10 test shard contents,
A/C capacity, and A/C resume parity before training. Training uses 200,000
samples, batch 32, at most 10,000 steps, shared 1,000-sample validation every
1,000 steps and patience 5 (min_delta 0.001). A and C run sequentially on one
GPU. This is a pilot subset audit, not a full 1M-corpus audit.

A running setup stage is not a training result. Colab runtime interruption can
stop the process; training saves checkpoints directly to Drive every 1,000
steps and at validation boundaries. With the same source, config and local
data path, relaunching resumes a committed checkpoint. Runtime preflight and
environment setup must be revalidated after a runtime replacement; do not
blindly reuse persistent setup markers in a fresh VM.

## Preflight outcome (12:13 local)

Pinned setup passed in 112 s; native bf16 passed; 182 tests passed in 56.79 s.
220 transferred files passed checksums. Formal training stopped at the strict
data audit due to assigned/stored stereo disagreement at train_00011370.
Independent capacity process 6590 completed both A/C 50-step bf16 probes
(exit 0). Peak reserved memory: A 3.053 GB, C 4.886 GB. Throughput on synthetic
optimizer inputs: A 518.17, C 100.60 images/s; excludes data I/O and validation.
Diagnostic process 6589 scans all transferred pairs and writes
reports/data-audit-diagnostic.json; it never writes data-audit.ok.

## Clean derived pilot

The complete 215,996-pair diagnostic found exactly three train stereo mismatches:
train_00011370, train_00057670, train_00167450; none in val/test. Four additional
train records were already absent from rendering within this transferred prefix.
Original files are preserved. A derived view removes only the three mismatched
PNG/JSON pairs and corresponding manifest rows, fills the train prefix to
200,000, and leaves every validation/test byte and selected identity unchanged.

Run ID: `colab-20260930-ac-s0-clean3`; pipeline PID `8579`.
Persistent results: `MyDrive/ouroboros/colab-20260930-ac-s0/clean3/`.
Strict audit passed in 258.12 s. A resume comparison passed over 100 steps
with zero mean relative loss difference. C comparison is running at this update.
After both comparisons pass, the same process runs A and then C seed-0 pilots.
`clean3/reports/quarantine.json` contains original/derived archive hashes and
the excluded labels. The original failed audit remains recorded as failed.

## Formal training verified

Both A/C resume comparisons passed with zero mean relative loss difference over
100 compared steps; their training and validation data identities match.
Pipeline PID 8579 is running A seed 0, observed at step 1,000, training loss
0.861359, last 50-step training throughput 451.97 images/s. These are training
metrics, not held-out accuracy. C seed 0 follows automatically after A.
The original source archive plus the clean-helper SHA256 in clean3-launch.json
fully identify the running code and the three-record quarantine operation.

## Training complete — checked 2026-09-30 21:34 PDT

Both seed-0 runs completed 10,000 steps, exit 0; final and validation-best
checkpoints persist in Drive. Shared 1,000-example validation exact match:
A 0.668, C 0.635. Both selected checkpoints are step 10,000. Training stage
wall times: A 959.95 s, C 3688.91 s. These are validation results from one seed,
not held-out test or rotation results. Full training logs and configs copied
to status-update/colab/clean3/{A,C}-{log.jsonl,config.yaml}; pipeline-status.json
now records completion. Evaluation had not started at this observation.

Hoffman2 container setup 14981448 and GPU probe 14981450 completed exit 0.
Expanse SSH control socket no longer exists and fresh authentication is denied;
its geometry array's latest status could not be verified.
