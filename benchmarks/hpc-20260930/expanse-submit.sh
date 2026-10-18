set -euo pipefail
task_root=/expanse/lustre/scratch/yguan1/temp_project
test -d "$task_root"
test -w "$task_root"
task_dir=$(mktemp -d "$task_root/ouroboros-probe.XXXXXX")
cd "$task_dir"
cat > job.sh <<'OUROBOROS_JOB_EOF'
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
OUROBOROS_JOB_EOF
sbatch --account=cla175 job.sh
printf 'logs=%s
' "$task_dir"
