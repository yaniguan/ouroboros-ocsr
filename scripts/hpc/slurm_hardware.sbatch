#!/bin/bash
#SBATCH --job-name=ouroboros_gpu_probe
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
# Supply --account with a verified existing project allocation at submission.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch, not on a login node}"
printf 'job_id=%s\n' "$SLURM_JOB_ID"
hostname
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
