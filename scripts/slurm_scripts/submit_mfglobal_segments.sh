#!/bin/bash
# Submit the Seq-Global tuning as a chain of 5-hour resumable segments.
# Usage: ./submit_mfglobal_segments.sh [n_segments]
# Change ACCOUNT here only. Existing class-account jobs are not migrated.
set -euo pipefail
ACCOUNT="${ACCOUNT:-plgoncotherapy2-gpu-a100}"
TEAM="${TEAM:-plggfastsim}"
SEGMENTS="${1:-4}"
cd "$(dirname "$0")/../.." || exit 1

# Refuse to create a second writer chain for the same Optuna studies.
if command -v squeue >/dev/null 2>&1; then
    if squeue -u "${USER}" -h -o "%j" | grep -Eq '^BBP_MFGlobal(Tune|Seg)'; then
        echo "A Seq-Global tuning job is already queued/running; stop it deliberately before starting a second writer chain." >&2
        exit 1
    fi
fi

PREV=""
for i in $(seq 1 "$SEGMENTS"); do
    if [ -z "$PREV" ]; then
        JOBID=$(sbatch --account="$ACCOUNT" --export=ALL,SEGMENT_INDEX="$i" \
            scripts/slurm_scripts/mfglobal_tune_segment.slurm | awk '{print $4}')
    else
        JOBID=$(sbatch --account="$ACCOUNT" --export=ALL,SEGMENT_INDEX="$i" \
            --dependency="afterany:${PREV}" \
            scripts/slurm_scripts/mfglobal_tune_segment.slurm | awk '{print $4}')
    fi
    echo "segment ${i}: ${JOBID} (account ${ACCOUNT}, team ${TEAM})"
    PREV="$JOBID"
done
