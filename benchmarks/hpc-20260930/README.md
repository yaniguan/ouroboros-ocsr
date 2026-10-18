# Cluster hardware preflight — 2026-09-30

This section records the initial hardware-only submission. The subsequently
authorized training/geometry wave and applied validation patch are recorded in
[gpu-wave.md](gpu-wave.md).

Both SSH connections and existing writable scratch paths were verified before
submission. Existing resource access only; no purchases, software installation,
dataset transfer or training occurred during that initial probe. The
shared-validation patch was still unapplied at that point.

| Cluster | Job ID | Verified access | Latest observed state |
| --- | --- | --- | --- |
| Hoffman2 | 14974099 | `sautet` group; job project `sautet_prj` | `qw`, requesting one A100, 2 GB RAM, 5 minutes |
| Expanse | 54534179 | GPU allocation `cla175` / `TG-CHE170060`, state `allow`, AVAILABLE 155000 | COMPLETED, exit `0:0`, elapsed 1 second; one V100 32 GB |

Hoffman2 scratch: `/u/scratch/y/yaniguan` (writable; filesystem reported 641 TB free).
Job directory: `/u/scratch/y/yaniguan/ouroboros-probe.wXcah6`.
The first submission was rejected by the job submission verifier because inherited
`C.UTF-8` is unsupported. The retained retry sets `LC_ALL=C LANG=C` before `qsub`.
It was accepted; no Job ID was issued for the rejected attempt.

Expanse scratch: `/expanse/lustre/scratch/yguan1/temp_project` (writable; filesystem
reported 1.2 PB free). The supplied `/scratch/expanse/...` spelling does not exist.
Job directory: `/expanse/lustre/scratch/yguan1/temp_project/ouroboros-probe.XlmOJQ`.
The job ran on `exp-1-58`, reporting Tesla V100-SXM2-32GB, 32768 MiB and driver
580.82.07. Before/after allocation queries showed USED 2 → 3 and USED BY PROJECT
47018 → 47019; AVAILABLE remained 155000. Raw output is retained in the receipts.

SSH uses existing authenticated multiplex sessions: configured `expanse` alias,
and `yaniguan@hoffman2.idre.ucla.edu`. A fresh batch SSH connection to Expanse
failed authentication, while the existing session worked. No credentials changed.

This is a hardware-only `nvidia-smi` check, not model imports, CUDA training,
native-bf16, resume, dataset integrity or scientific accuracy validation.
The shared-validation patch passed `git apply --check` against local HEAD `5f9efc0`;
no patch application or new test run was performed during this submission task.

## How to run

Inspect existing jobs (these commands do not submit duplicates):

```bash
ssh yaniguan@hoffman2.idre.ucla.edu 'export LC_ALL=C LANG=C; qstat -j 14974099'
ssh yaniguan@hoffman2.idre.ucla.edu 'export LC_ALL=C LANG=C; qacct -j 14974099'
ssh expanse 'sacct -j 54534179 -X --format=JobID,State,Elapsed,ExitCode,AllocTRES'
ssh expanse 'cat /expanse/lustre/scratch/yguan1/temp_project/ouroboros-probe.XlmOJQ/slurm-54534179.out'
```

Submission scripts and their stdout/stderr are retained beside this file. The
`*-submit*.sh` files create and submit new jobs if rerun; use status commands above
to follow these jobs. Templates are under `scripts/hpc/`.

Scheduler syntax checked against official documentation:
[Hoffman2 production Grid Engine](https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing-Altair-production.html)
and [Expanse user guide](https://www.sdsc.edu/systems/expanse/user_guide.html).
Current account, path and job facts come from the retained live cluster responses.
