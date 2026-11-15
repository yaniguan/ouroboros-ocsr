# GPU preflight and geometry checks — 2026-09-30

Summary files from the initial GPU capacity and MACE-OFF23 geometry runs:

- `capacity-A.json`: 50-step bf16 synthetic throughput and memory check on A100-40GB for arm A (518.2 images/s, 3.05 GB peak reserved).
- `capacity-C.json`: 50-step bf16 synthetic throughput and memory check on A100-40GB for arm C (100.6 images/s, 4.89 GB peak reserved).
- `mace-cuda.json`: single-molecule MACE-OFF23 CUDA relaxation check on V100-32GB (14.3 s, converged).
- `enantiomers.json`: 20 chiral molecules and their mirror conformers (`x -> -x`) relaxed with MACE-OFF23 (`fmax = 0.05 eV/A`); all 20 pairs converged with `max |dE| = 6.48e-10 eV`.

Full A/B/C pilot training and 24-angle evaluation summaries are in [`benchmarks/v0.1.0-alpha/`](../v0.1.0-alpha/) and [`docs/v0-results.md`](../../docs/v0-results.md).
