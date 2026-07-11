#!/bin/bash
# Phase 4 hyperparameter search on plgrid-now.
#
# Same shape as the Phase 3 ladder runner, but the study list comes from a spec
# file rather than a hard-coded array, because Phase 4 tunes three campaigns with
# different budgets (FF at 50 trials, CaFo and the CNN baselines at 30, the BP MLP
# baselines at 50) and they are submitted separately.
#
# Study names are prefixed phase4_ so they can never top up a ladder_* study.
# Two jobs must never open the same SQLite study: one file, two nodes and Lustre
# is how a study gets corrupted. Drive this with one pilot in flight.
#
#SBATCH --job-name=P4_hpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --output=slurm_logs/phase4/%x-%j.out
#SBATCH --error=slurm_logs/phase4/%x-%j.err

SPEC_FILE=${PHASE4_HPO_SPEC:?set PHASE4_HPO_SPEC to a spec file listing tuning configs}
BUDGET_SEC=${PHASE4_HPO_BUDGET:-2700}
# A trial is only started with this much budget left, otherwise the trial in
# flight overruns the one-hour cap and its result is thrown away.
MIN_SLICE_SEC=${PHASE4_HPO_MIN_SLICE:-600}

cd "$SLURM_SUBMIT_DIR" || exit 1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_MODE=offline

mapfile -t TUNE_CONFIGS < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$SPEC_FILE" | grep -v '^$')
NUM_STUDIES=${#TUNE_CONFIGS[@]}

START_IDX=${PHASE4_HPO_START:-1}
END_IDX=${PHASE4_HPO_END:-$NUM_STUDIES}

count_trials() {
    python - "$1" <<'PY'
import os, sys, optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
name = sys.argv[1]
db = os.path.join("results/optuna", name + ".db")
if not os.path.exists(db):
    print(0)
    raise SystemExit
try:
    study = optuna.load_study(study_name=name, storage="sqlite:///" + db)
except Exception:
    print(0)
    raise SystemExit
print(sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE))
PY
}

target_trials() {
    python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('tuning',{}).get('n_trials',20))" "$1"
}

echo "#####################################################################"
echo "Phase 4 HPO on plgrid-now"
echo "Spec: ${SPEC_FILE} (${NUM_STUDIES} studies), range ${START_IDX}-${END_IDX}"
echo "Budget: ${BUDGET_SEC}s   min slice: ${MIN_SLICE_SEC}s"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "$(date)"
echo "#####################################################################"

START_TS=$(date +%s)
ADDED=0
SKIPPED=0
FAILED=0
REMAINING=0

for (( IDX = START_IDX; IDX <= END_IDX; IDX++ )); do
    CONFIG_FILE=${TUNE_CONFIGS[$((IDX - 1))]}
    if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
        echo "--- ERROR: spec line ${IDX} is not a readable config ('${CONFIG_FILE}')"
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    STUDY_NAME="phase4_$(basename "$CONFIG_FILE" .yaml)"
    TARGET=$(target_trials "$CONFIG_FILE")
    DONE=$(count_trials "$STUDY_NAME")
    SHORT=$(( TARGET - DONE ))

    if [ "$SHORT" -le 0 ]; then
        echo "--- ${STUDY_NAME}: ${DONE}/${TARGET}, nothing to do"
        SKIPPED=$(( SKIPPED + 1 ))
        continue
    fi

    ELAPSED=$(( $(date +%s) - START_TS ))
    SLICE=$(( BUDGET_SEC - ELAPSED ))
    if [ "$SLICE" -lt "$MIN_SLICE_SEC" ]; then
        echo "--- ${STUDY_NAME}: ${DONE}/${TARGET}, only ${SLICE}s left; next chunk takes it"
        REMAINING=$(( REMAINING + SHORT ))
        continue
    fi

    echo "--- ${STUDY_NAME}: ${DONE}/${TARGET}, spending up to ${SLICE}s"
    python scripts/run_optuna_search.py \
        --config "$CONFIG_FILE" \
        --study-name "$STUDY_NAME" \
        --timeout "$SLICE" || FAILED=$(( FAILED + 1 ))

    NOW=$(count_trials "$STUDY_NAME")
    ADDED=$(( ADDED + NOW - DONE ))
    if [ "$TARGET" -gt "$NOW" ]; then
        REMAINING=$(( REMAINING + TARGET - NOW ))
    fi
    echo "--- ${STUDY_NAME}: now ${NOW}/${TARGET}"
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo "#####################################################################"
echo "HPO_NOW_SUMMARY added=${ADDED} remaining=${REMAINING} elapsed=${TOTAL_ELAPSED}s"
# The driver counts a pilot as progress only if completed>0, and stops when both
# completed and failed are 0, so trials added is what it needs to read here.
echo "PILOT_SUMMARY completed=${ADDED} skipped=${SKIPPED} failed=${FAILED} elapsed=${TOTAL_ELAPSED}s"
echo "$(date)"
echo "#####################################################################"

deactivate
