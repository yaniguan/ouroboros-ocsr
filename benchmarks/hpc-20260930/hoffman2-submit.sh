set -euo pipefail
task_root=/u/scratch/y/yaniguan
test -d "$task_root"
test -w "$task_root"
task_dir=$(mktemp -d "$task_root/ouroboros-probe.XXXXXX")
cd "$task_dir"
cat > job.sh <<'OUROBOROS_JOB_EOF'
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N ouroboros_gpu_probe
#$ -l gpu,A100,cuda=1,h_rt=00:05:00,h_data=2G
set -euo pipefail
: "${JOB_ID:?Submit through qsub, not on a login node}"
printf 'job_id=%s\n' "$JOB_ID"
hostname
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
OUROBOROS_JOB_EOF
qsub job.sh
printf 'logs=%s
' "$task_dir"
