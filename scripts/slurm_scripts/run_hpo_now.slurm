#!/bin/bash
# Phase 3, task 7: finish the remaining ladder HPO studies on plgrid-now.
#
# Same study list and study names as run_ladder_hpo.slurm, so both scripts top up
# the same SQLite databases. They must never run at the same time: two nodes
# writing one SQLite file over Lustre is how a study gets corrupted. The driver
# drains the plgrid-gpu-a100 queue before it starts.
#
# plgrid-now caps a job at one hour, so a chunk spends BUDGET seconds walking
# HPO_NOW_INDICES in order and stops starting work with enough slack for the
# trial in flight to land. Studies that already hit their target no-op.
#
#SBATCH --job-name=H_now
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --output=slurm_logs/phase3/%x-%j.out
#SBATCH --error=slurm_logs/phase3/%x-%j.err

TUNE_CONFIGS=(
    # --- Rung 1: BP ---
    "configs/tuning/bp_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bp_fashion_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bp_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/bp_cifar100_mlp_3x2000_tune.yaml"

    # --- Rung 2: BP-DS ---
    "configs/tuning/bpds_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bpds_fashion_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bpds_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/bpds_cifar100_mlp_3x2000_tune.yaml"

    # --- Rung 3: MF-Joint ---
    "configs/tuning/mfjoint_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/mfjoint_fashion_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/mfjoint_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/mfjoint_cifar100_mlp_3x2000_tune.yaml"

    # --- Rung 4: MF ---
    "configs/tuning/mf_mnist_mlp_2x1000_mf_tune.yaml"
    "configs/tuning/mf_fashion_mnist_mlp_2x1000_mf_tune.yaml"
    "configs/tuning/mf_cifar10_mlp_3x2000_mf_tune.yaml"
    "configs/tuning/mf_cifar100_mlp_3x2000_mf_tune.yaml"
)

# Cheapest first, so the studies a few trials short finish inside one chunk and
# the whole remaining budget lands on the two expensive MF studies.
INDICES_CSV=${HPO_NOW_INDICES:-7,8,4,15,16}
BUDGET_SEC=${HPO_NOW_BUDGET:-2700}
# An MF CIFAR trial costs ~6.5 min, so a study is only started with that much
# budget left; otherwise the trial in flight would overrun the one-hour cap.
MIN_SLICE_SEC=${HPO_NOW_MIN_SLICE:-450}

cd "$SLURM_SUBMIT_DIR" || exit 1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_MODE=online

# Completed trials in a study, or 0 if it has never been created.
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
echo "Ladder HPO on plgrid-now"
echo "Indices: ${INDICES_CSV}   budget: ${BUDGET_SEC}s"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "$(date)"
echo "#####################################################################"

START_TS=$(date +%s)
ADDED=0
REMAINING=0

IFS=',' read -r -a INDICES <<< "$INDICES_CSV"

for TASK_ID in "${INDICES[@]}"; do
    INDEX=$((TASK_ID - 1))
    if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "${#TUNE_CONFIGS[@]}" ]; then
        echo "--- skipping out-of-range index ${TASK_ID}"
        continue
    fi

    CONFIG_FILE=${TUNE_CONFIGS[$INDEX]}
    STUDY_NAME="ladder_$(basename "$CONFIG_FILE" .yaml)"

    TARGET=$(target_trials "$CONFIG_FILE")
    DONE=$(count_trials "$STUDY_NAME")
    SHORT=$(( TARGET - DONE ))

    if [ "$SHORT" -le 0 ]; then
        echo "--- ${STUDY_NAME}: ${DONE}/${TARGET}, nothing to do"
        continue
    fi

    ELAPSED=$(( $(date +%s) - START_TS ))
    SLICE=$(( BUDGET_SEC - ELAPSED ))
    if [ "$SLICE" -lt "$MIN_SLICE_SEC" ]; then
        echo "--- ${STUDY_NAME}: ${DONE}/${TARGET}, only ${SLICE}s left; leaving it for the next chunk"
        REMAINING=$(( REMAINING + SHORT ))
        continue
    fi

    echo "--- ${STUDY_NAME}: ${DONE}/${TARGET}, spending up to ${SLICE}s"
    python scripts/run_optuna_search.py \
        --config "$CONFIG_FILE" \
        --study-name "$STUDY_NAME" \
        --timeout "$SLICE"

    NOW=$(count_trials "$STUDY_NAME")
    ADDED=$(( ADDED + NOW - DONE ))
    REMAINING=$(( REMAINING + (TARGET > NOW ? TARGET - NOW : 0) ))
    echo "--- ${STUDY_NAME}: now ${NOW}/${TARGET}"
done

WANDB_LOCAL_DIR="${WANDB_DIR:-$SLURM_SUBMIT_DIR}/wandb"
if compgen -G "$WANDB_LOCAL_DIR/offline-run-*" > /dev/null; then
    wandb sync "$WANDB_LOCAL_DIR"/offline-run-* || echo "WARNING: wandb sync failed"
fi

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo "#####################################################################"
echo "HPO_NOW_SUMMARY added=${ADDED} remaining=${REMAINING} elapsed=${TOTAL_ELAPSED}s"
echo "$(date)"
echo "#####################################################################"

deactivate
