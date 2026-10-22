#!/bin/bash
# Prepared, guarded migration of the two pending class-account jobs to oncology.
# Default is a dry run. Set CONFIRM=YES only after the author approves.
set -euo pipefail

ACCOUNT="${ACCOUNT:-plgoncotherapy2-gpu-a100}"
JOBS=(3386582 3386586)
OLD_SCRIPTS=(
    "scripts/slurm_scripts/mfglobal_tune.slurm"
    "scripts/slurm_scripts/w1_cap_pilot.slurm"
)

cd "$(dirname "$0")/../.." || exit 1

echo "Pending jobs: ${JOBS[*]}"
echo "Target account: $ACCOUNT"
echo "Scripts: ${OLD_SCRIPTS[*]}"
echo

if [ "${CONFIRM:-NO}" != "YES" ]; then
    cat <<'EOF'
DRY RUN. The commands that would run are:

  squeue -j 3386582,3386586 -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
  sacct -j 3386582,3386586 --format=JobID,JobName%24,State,Elapsed,AllocTRES%40
  scancel 3386582 3386586
  while squeue -h -j 3386582,3386586 | grep -q .; do sleep 10; done
  sbatch --account=plgoncotherapy2-gpu-a100 scripts/slurm_scripts/mfglobal_tune.slurm
  sbatch --account=plgoncotherapy2-gpu-a100 scripts/slurm_scripts/w1_cap_pilot.slurm
  squeue -u "$USER" -o "%.18i %.20j %.20P %.20a %.8T %.10M %R"

Set CONFIRM=YES to execute. Do not do that without the author's explicit
approval; both old jobs currently have 0 GPU-hours and remain a historical
record on the class account until then.
EOF
    exit 0
fi

squeue -j "${JOBS[0]},${JOBS[1]}" -o "%.18i %.9P %.20j %.8T %.10M %.6D %R" || true
sacct -j "${JOBS[0]},${JOBS[1]}" --format=JobID,JobName%24,State,Elapsed,AllocTRES%40 || true
scancel "${JOBS[@]}"

while squeue -h -j "${JOBS[0]},${JOBS[1]}" | grep -q .; do
    sleep 10
done

for script in "${OLD_SCRIPTS[@]}"; do
    sbatch --account="$ACCOUNT" "$script"
done

squeue -u "$USER" -o "%.18i %.20j %.20P %.20a %.8T %.10M %R"
