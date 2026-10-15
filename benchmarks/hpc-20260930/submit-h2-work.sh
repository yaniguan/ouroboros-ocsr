set -euo pipefail
export LC_ALL=C LANG=C
cd /u/scratch/y/yaniguan/ouroboros-20260930
tar -xzf work.tar.gz -C code
test ! -e reports/submissions.tsv
: > reports/submissions.tsv
submit() {
  task=$1
  shift
  result=$(qsub -terse -cwd -S /bin/bash -N "ouro_$task" -l gpu,L40S,cuda=1,h_data=4G -pe shared 4 -j y -o reports -v "OURO_ROOT=/u/scratch/y/yaniguan/ouroboros-20260930,OURO_TASK=$task" "$@" code/scripts/hpc/work.sh)
  id=${result%%.*}
  printf '%s\t%s\n' "$task" "$result" >> reports/submissions.tsv
  printf '%s' "$id"
}
tests=$(submit tests -hold_jid 14974162 -l h_rt=02:00:00)
audit=$(submit audit -h -hold_jid 14974162 -l h_rt=02:00:00)
capacity=$(submit capacity -hold_jid "$tests" -t 1-5 -l h_rt=01:00:00)
resume=$(submit resume -hold_jid "$audit,$capacity" -t 1-2 -l h_rt=02:00:00)
pilot=$(submit pilot -hold_jid "$resume" -t 1-2 -l h_rt=12:00:00)
train=$(submit train -hold_jid "$pilot" -t 1-4 -l h_rt=12:00:00)
evaluate=$(submit evaluate -hold_jid "$pilot,$train" -t 1-6 -l h_rt=12:00:00)
cat reports/submissions.tsv
