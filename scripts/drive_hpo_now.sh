#!/bin/bash
# Phase 3, task 7: finish the remaining ladder HPO studies on plgrid-now.
#
# Waits for the ladder driver to finish first, because plgrid-now allows one job
# per user and the 240-run ladder is the higher priority. Then it drains the
# plgrid-gpu-a100 HPO queue, so no chunk there can wake up and write the same
# Optuna SQLite file from a second node, and tops the studies up one hour at a
# time.
#
# Run it detached from the login shell:
#   cd ~/BeyondBackpropagation
#   nohup setsid bash scripts/drive_hpo_now.sh > /dev/null 2>&1 < /dev/null &
#
# Stop it with: pkill -f drive_hpo_now.sh
set -u

REPO="$HOME/BeyondBackpropagation"
cd "$REPO" || exit 1

LOGDIR="$REPO/slurm_logs/phase3"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/hpo_now_driver.log"

INDICES=${HPO_NOW_INDICES:-7,8,4,15,16}
MAX_JOBS=${HPO_NOW_MAX_JOBS:-40}
DRAIN=${HPO_NOW_DRAIN:-1}
STALL_LIMIT=3

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "driver starting: studies ${INDICES}, at most ${MAX_JOBS} chunks"

# --- 1. yield the single plgrid-now slot to the ladder ---------------------
while pgrep -f 'drive_ladder_now\.sh' > /dev/null 2>&1; do
    sleep 120
done
log "ladder driver has exited"

while squeue -h -u "$USER" -p plgrid-now 2>/dev/null | grep -q .; do
    sleep 60
done
log "plgrid-now queue is clear"

# --- 2. make sure nothing else can touch these studies ---------------------
if [ "$DRAIN" = "1" ]; then
    VICTIMS=$(squeue -h -u "$USER" -p plgrid-gpu-a100 -o "%i %j" | awk '$2 ~ /^H_/ {print $1}')
    if [ -n "$VICTIMS" ]; then
        log "cancelling $(wc -w <<< "$VICTIMS") queued plgrid-gpu-a100 HPO chunks: $(tr '\n' ' ' <<< "$VICTIMS")"
        scancel $VICTIMS
        sleep 10
    else
        log "no plgrid-gpu-a100 HPO chunks to cancel"
    fi
fi

# --- 3. top the studies up, one hour at a time -----------------------------
stall=0
for (( i = 1; i <= MAX_JOBS; i++ )); do
    JID=$(sbatch --parsable \
        --export=ALL,HPO_NOW_INDICES="$INDICES" \
        scripts/slurm_scripts/run_hpo_now.slurm 2>&1)

    if ! [[ "$JID" =~ ^[0-9]+$ ]]; then
        log "submit rejected (${JID//$'\n'/ }); retrying in 120s"
        sleep 120
        continue
    fi
    log "chunk ${i} submitted as ${JID}"

    while squeue -h -j "$JID" 2>/dev/null | grep -q .; do
        sleep 60
    done

    SUMMARY=$(grep -h HPO_NOW_SUMMARY "$LOGDIR/H_now-${JID}.out" 2>/dev/null | tail -1)
    log "chunk ${JID} finished: ${SUMMARY:-<no summary written>}"

    ADDED=$(sed -n 's/.*added=\(-\{0,1\}[0-9]\{1,\}\).*/\1/p' <<< "$SUMMARY")
    LEFT=$(sed -n 's/.*remaining=\([0-9]\{1,\}\).*/\1/p' <<< "$SUMMARY")

    if [ "${LEFT:-}" = "0" ]; then
        log "every study reached its trial target. Done."
        exit 0
    fi

    if [ -z "${ADDED:-}" ] || [ "${ADDED:-0}" -le 0 ]; then
        stall=$(( stall + 1 ))
        log "no trials added (${stall}/${STALL_LIMIT} consecutive)"
        if [ "$stall" -ge "$STALL_LIMIT" ]; then
            log "stopping: no progress across ${STALL_LIMIT} chunks. Inspect $LOGDIR."
            exit 1
        fi
    else
        stall=0
    fi
done

log "reached MAX_JOBS=${MAX_JOBS}; stopping."
