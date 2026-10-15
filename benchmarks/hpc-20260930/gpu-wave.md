# GPU work wave — 2026-09-30

## Later status update

Expanse setup 54534206, preparation 54534185, CUDA probe 54534192 and enantiomer
job 54534193 all completed with exit 0. Geometry tests: 31 passed in 5.74 s.
The V100 alanine optimization converged in 14.30 s (job elapsed 9m15s including
startup). All 20 enantiomer pairs converged; maximum |delta E| was
6.475602276623249e-10 eV, passing the 1e-3 criterion. Array 54534186 had task 0
running and 19 tasks pending priority; 3 molecule records were processed at inspection.

Hoffman2 setup 14974162 failed immediately with exit 141: `head -1` closed the
`ldd --version` pipe and pipefail treated SIGPIPE as fatal. Downstream jobs
14974182–14974189 reached their prerequisite guards and exited without training.
They are historical failed submissions, not active/completed experiments.
The pipeline was replaced with full `ldd --version` output; error traps now report
task, job, line and exit code. The actual A100 node has glibc 2.17, below the
requirement of the modern pinned stack. Site Apptainer/Singularity currently
needs an `unsquashfs` helper in PATH to open its supplied SIF images.

Repair job **14981448** installs that helper in a private scratch prefix, checks
site containers for a modern glibc and Fortran compiler, and attempts the pinned
environment inside the selected container. GPU probe **14981450** depends on it
and checks the current model on a scheduler-bound A100. Neither repair success
nor resumed training is claimed yet. The long training arrays have not been
resubmitted pending this compatibility check.

Evidence: `status-update/hoffman2-accounting.txt`, `mace-cuda.json`,
`geometry-tests.log`, `enantiomers.json`, and `repair-submission.txt`.
Container route follows the site's [software documentation](https://www.hoffman2.idre.ucla.edu/Using-H2/Software/Software.html).

The sections below preserve the original submission plan and IDs.

User authorized broad use of existing Hoffman2 and Expanse GPU allocations,
without a GPU-saving constraint. No purchases or unrelated jobs were changed.
This wave submits **41 single-GPU work tasks**, plus setup, test and audit jobs.
Submission and dependency waits are not evidence of model or scientific success.

## Submitted work

| Cluster | Job / array | Task count | Work |
| --- | --- | --- | --- |
| Hoffman2 | 14974185.1-5 | 5 | A/B/C/C+/D, 50 optimizer steps, bf16, batch 32, 384 px, sequence 128 |
| Hoffman2 | 14974186.1-2 | 2 | A/C, 100-step resumed vs uninterrupted loss comparison |
| Hoffman2 | 14974187.1-2 | 2 | A/C synthetic-only 200k, seed 0 |
| Hoffman2 | 14974188.1-4 | 4 | A/C, seeds 1 and 2, after both seed-0 pilots finish |
| Hoffman2 | 14974189.1-6 | 6 | Full 10k synthetic test and 24-angle rotation sweep (2k prefix), committed validation-best checkpoint |
| Expanse | 54534192 | 1 | MACE-OFF23 medium CUDA/float64 alanine relaxation probe |
| Expanse | 54534186_0-19 | 20 | Fixed first 1,000 synthetic test molecules, disjoint 50-row batches, 10 conformers each |
| Expanse | 54534193 | 1 | 20 enantiomer-pair energy checks with mirrored initial conformers |

Hoffman2 prerequisites: setup **14974162**, full pytest **14974182**, data audit
**14974183**. Every work request uses one A100, 4 CPU slots and 16 GB host RAM.
Capacity time limit is 1 hour; resume 2 hours; training and evaluation 12 hours.
The array indices map to A,C alternating, then successive paired seeds.

Expanse account is `cla175` / `TG-CHE170060`; geometry preparation **54534185**
downloads one shared medium-model file, hashes it and runs geometry unit tests.
Preparation requires setup job **54534206** to succeed (see `submitted-jobs.json`).
CUDA probe follows preparation; the 20-task array and enantiomer job follow the
CUDA probe. They request the standard `gpu-shared` partition, one GPU, 2 CPUs and
16 GB RAM. Limits: probe 30 min, geometry 12 h/task, enantiomers 2 h.

## Protocol and data

The shared-validation patch is applied, excluding its stale TASK.md hunk. A/C
use identical batch 32, maximum 10,000 steps, fixed first 1,000 val samples,
validation every 1,000 steps, patience 5 and min_delta 0.001. The strict best score
chooses the checkpoint even when a small gain does not reset patience. Numeric
settings, vocabulary and decoder are unchanged between arms.

Resume checks deliberately use only 32 validation samples every 25 steps, with
patience 20, to exercise checkpointed validation state across the 100 compared
steps. This is a resume diagnostic, not the training/selection protocol.

The corpus transfer includes 201 train shards (enough to select 200,000 usable
images after recorded rendering drops), all 5 validation and 10 test shards,
and all original manifests. Files total 1,613,546,004 bytes. The transfer tar is
1,613,895,680 bytes, SHA256
`87134b1dc56e14a63c3ae5072692636fd756c67411c07ee8dff008a7f698e00b`;
the remote checksum passed. Per-file hashes are in `data-transfer-receipts.json`.
The explicit upload hold on audit job 14974183 was released after extraction;
its setup dependency remains.

The audit checks every transferred PNG/JSON pair, PNG integrity, assigned-label
vs stored-label canonical stereo identity, key14, duplicates and split overlap.
Missing render records remain explicit, never successes. It freezes content/order
identities for the 200k train, 1k validation and 10k test selections. This is a
subset audit, not an exhaustive audit of the full 1M corpus. Real pseudo-labels
are not included in these training/evaluation jobs.

Geometry uses the retained assigned-stereo synthetic test manifest, MACE-OFF23
medium with a recorded weight SHA256, float64, seed 0, 10 requested conformers,
fmax 0.05 eV/angstrom and at most 500 LBFGS steps per conformer. Per-molecule
atomic cache records keep failures and permit reruns. Each task has denominator
50; `all_10_converged` additionally requires all 10 requested conformers. These
are geometry baselines, not energy propagation from OCSR predictions.

## Environments and repairs

- Base revision `5f9efc07fa9530d351296b635918fb8d7c8543bc`, modified by the
  validation patch and HPC scripts. `final-source.json` and remote
  `code/hpc-source.json` record file hashes. No commit/push was performed.
- Local focused regression: 14 passed in 10.48 s. After adding native-bf16
  rejection coverage, 4 benchmark tests passed in 10.90 s. Changed Python Ruff
  and shell syntax checks pass. These use the existing local environment and
  are not a pinned cluster test pass.
- Cluster dependencies are exactly `requirements.txt`; private venvs/cache under
  each project scratch root. No existing user environments are modified.
- Expanse setup 54534182 was cancelled when managed-Python network download
  stalled. Setup 54534188 used system Python 3.11.5 but failed building py3nj
  because Python development headers were absent. Replacement stages official
  CPython 3.11.16 with headers and reuses downloaded wheels. The failed venv is
  retained as `venv-system311-failed`; dependents are redirected to the replacement.
  Replacement setup job is **54534206**; installation completion remains pending.
- Staged CPython archive SHA256:
  `f05084f3fbd7a0c7ed762ea6fd2455976a463952bb5c3253fc6cdc777f84728d`.
- The final 289-file code snapshot was rehashed on both clusters: zero mismatches.
- Expanse's submit plugin rejected `--gpus=v100:1`; the accepted array uses the
  documented `--gpus=1`. No job was created by the rejected submission.
- Hoffman2 requests use `LC_ALL=C LANG=C`. Initial L40S requests were rerouted
  to `gpu_a100.q` after checking access/resource constraints. A100 validation
  reported a possible assignment with 4 slots. No access controls were changed.
- SGE dependencies alone do not require predecessor success; explicit success
  marker checks in each task prevent training on a failed prerequisite. Slurm
  uses `afterok` dependencies. Work guards reject unavailable CUDA and emulated
  bf16 for OCSR. No precision fallback is permitted.

## How to run

Working roots:

- Hoffman2: `/u/scratch/y/yaniguan/ouroboros-20260930`
- Expanse: `/expanse/lustre/scratch/yguan1/temp_project/ouroboros-20260930`

Inspect without submitting duplicates:

```bash
ssh yaniguan@hoffman2.idre.ucla.edu 'export LC_ALL=C LANG=C; qstat -u yaniguan | grep ouro'
ssh yaniguan@hoffman2.idre.ucla.edu 'export LC_ALL=C LANG=C; qstat -j 14974187'
ssh expanse 'squeue -u yguan1'
ssh expanse 'sacct -j 54534185,54534186,54534192,54534193 --format=JobID,State,Elapsed,ExitCode,AllocTRES'
```

Logs are under each root's `reports/`. OCSR checkpoints and predictions are under
Hoffman2 `runs/`; per-molecule geometry records under Expanse `geometry/part-*/`.
The enantiomer summary is Expanse `code/benchmarks/geometry/enantiomers.json`.
Before retrying any failed job, inspect its log and preserve its failure evidence.
Resume training with the same config and output directory; geometry cache permits
rerunning the same task index. Capacity reports intentionally reject overwrite.
