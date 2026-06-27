#!/bin/bash
# Phase 3, task 8: keeps exactly one plgrid-now pilot job in flight.
#
# plgrid-now allows MaxSubmitJobsPU=1, so neither an array nor a dependency
# chain is possible: a second sbatch is rejected outright with
# QOSMaxSubmitJobPerUserLimit. The only way to use the partition continuously is
# to submit, wait, and submit again, which is what this loop does.
#
# Run it detached from the login shell:
#   cd ~/BeyondBackpropagation
#   nohup scripts/drive_ladder_now.sh > /dev/null 2>&1 &
#
# Stop it with: pkill -f drive_ladder_now.sh
set -u

REPO="$HOME/BeyondBackpropagation"
cd "$REPO" || exit 1

# Must match the runner's #SBATCH --output directory: the driver reads each pilot's
# PILOT_SUMMARY out of that file to decide whether to submit another.
LOGDIR=${LADDER_NOW_LOGDIR:-$REPO/slurm_logs/phase3}
mkdir -p "$LOGDIR"

# Which pilot to drive. The top-up runner (task 11) has the same submit/wait/resubmit
# shape and the same PILOT_SUMMARY contract, so it only needs a different script name.
SCRIPT=${LADDER_NOW_SCRIPT:-scripts/slurm_scripts/run_ladder_now.slurm}
JOBNAME=${LADDER_NOW_JOBNAME:-L_now}
LOG="$LOGDIR/${JOBNAME}_driver.log"

START=${LADDER_NOW_START:-1}
END=${LADDER_NOW_END:-240}
MAX_JOBS=${LADDER_NOW_MAX_JOBS:-60}
STALL_LIMIT=3

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "driver starting: ${SCRIPT} indices ${START}-${END}, at most ${MAX_JOBS} pilots"

stall=0
for (( i = 1; i <= MAX_JOBS; i++ )); do
    JID=$(sbatch --parsable \
        --export=ALL,LADDER_NOW_START="$START",LADDER_NOW_END="$END",TOPUP_NOW_START="$START",TOPUP_NOW_END="$END" \
        "$SCRIPT" 2>&1)

    if ! [[ "$JID" =~ ^[0-9]+$ ]]; then
        log "submit rejected (${JID//$'\n'/ }); retrying in 120s"
        sleep 120
        continue
    fi
    log "pilot ${i} submitted as ${JID}"

    while squeue -h -j "$JID" 2>/dev/null | grep -q .; do
        sleep 60
    done

    SUMMARY=$(grep -h PILOT_SUMMARY "$LOGDIR/${JOBNAME}-${JID}.out" 2>/dev/null | tail -1)
    log "pilot ${JID} finished: ${SUMMARY:-<no summary written>}"

    DONE_N=$(sed -n 's/.*completed=\([0-9]\{1,\}\).*/\1/p' <<< "$SUMMARY")
    FAIL_N=$(sed -n 's/.*failed=\([0-9]\{1,\}\).*/\1/p' <<< "$SUMMARY")

    if [ "${DONE_N:-}" = "0" ] && [ "${FAIL_N:-}" = "0" ]; then
        log "every run in ${START}-${END} already has a summary JSON. Done."
        exit 0
    fi

    # An empty summary means the pilot died before reporting, which counts as
    # no progress; three in a row means something is broken, not merely slow.
    if [ -z "${DONE_N:-}" ] || [ "${DONE_N:-0}" = "0" ]; then
        stall=$(( stall + 1 ))
        log "no completed runs (${stall}/${STALL_LIMIT} consecutive)"
        if [ "$stall" -ge "$STALL_LIMIT" ]; then
            log "stopping: no progress across ${STALL_LIMIT} pilots. Inspect $LOGDIR."
            exit 1
        fi
    else
        stall=0
    fi
done

log "reached MAX_JOBS=${MAX_JOBS}; stopping."
