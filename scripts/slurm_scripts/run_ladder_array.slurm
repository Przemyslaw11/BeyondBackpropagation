#!/bin/bash
# Phase 3, task 8: the ablation ladder at n = 20 seeds.
#
# THE LADDER
#   1 BP              output_layer readout, no auxiliary losses, global gradients
#   2 BP-DS           output_layer readout, auxiliary losses,    global gradients
#   3 MF-Joint        M_L readout,          auxiliary losses,    global gradients
#   4 MF-recompute    M_L readout,          auxiliary losses,    local  gradients
#   5 MF-cache-device as rung 4, activations cached on the GPU
#   6 MF-cache-host   as rung 4, activations cached in host memory
#
# Each adjacent pair changes exactly one thing, so each step is attributable.
#
# WHY 20 RUNG-CONFIGS AND NOT 24
# Rungs 5 and 6 are absent on CIFAR-10 and CIFAR-100. CIFAR training applies
# RandomCrop and RandomHorizontalFlip, so an activation computed in one epoch is
# not the activation the next epoch needs; a cross-epoch cache would train on
# stale inputs. src/algorithms/mf.py refuses the combination, correctly. The
# ladder is therefore 2 configs x 6 rungs + 2 configs x 4 rungs = 20.
#
#   20 rung-configs x 20 seeds = 400 runs.
#
#SBATCH --job-name=BBP_Ladder
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=04:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-420

CONFIG_FILES=(
    # --- MNIST 2x1000: all six rungs ---
    "configs/bp_baselines/mnist_mlp_2x1000_bp.yaml"
    "configs/bp_ds/mnist_mlp_2x1000.yaml"
    "configs/mf_joint/mnist_mlp_2x1000.yaml"
    "configs/mf/mnist_mlp_2x1000.yaml"
    "configs/mf/mnist_mlp_2x1000_cache_device.yaml"
    "configs/mf/mnist_mlp_2x1000_cache_host.yaml"

    # --- Fashion-MNIST 2x1000: all six rungs ---
    "configs/bp_baselines/fashion_mnist_mlp_2x1000_bp.yaml"
    "configs/bp_ds/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf_joint/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000_cache_device.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000_cache_host.yaml"

    # --- CIFAR-10 3x2000: rungs 1-4 only, augmentation rules out the cache ---
    "configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml"
    "configs/bp_ds/cifar10_mlp_3x2000.yaml"
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"
    "configs/mf/cifar10_mlp_3x2000.yaml"

    # --- CIFAR-100 3x2000: rungs 1-4 only, same reason ---
    "configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml"
    "configs/bp_ds/cifar100_mlp_3x2000.yaml"
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"
    "configs/mf/cifar100_mlp_3x2000.yaml"

    # --- Task 10 diagnostic: rung 4 with the pre-Phase-2 stopping rule ---
    # Writes to results/diagnostics, not results/runs. Flat indices 401-420.
    "configs/diagnostics/cifar10_mlp_3x2000_legacy_es.yaml"
)

# Twenty seeds. The rungs share this list, which is what makes the analysis
# paired: one seed fixes the train/validation split and the initialisation.
SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61)

NUM_CONFIGS=${#CONFIG_FILES[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_JOBS=$((NUM_CONFIGS * NUM_SEEDS))
echo "INFO: ${NUM_CONFIGS} rung-configs x ${NUM_SEEDS} seeds = ${NUM_JOBS} runs."
if [ "${SLURM_ARRAY_TASK_MAX}" -ne "${NUM_JOBS}" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! WARNING: --array (1-${SLURM_ARRAY_TASK_MAX}) does not match ${NUM_JOBS} runs. !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

FLAT_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ $FLAT_INDEX -lt 0 ] || [ $FLAT_INDEX -ge $NUM_JOBS ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) for ${NUM_JOBS} runs." >&2
    exit 1
fi

INDEX=$((FLAT_INDEX / NUM_SEEDS))
SEED_INDEX=$((FLAT_INDEX % NUM_SEEDS))

CONFIG_FILE=${CONFIG_FILES[$INDEX]}
export EXPERIMENT_SEED=${SEEDS[$SEED_INDEX]}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found for task ID $SLURM_ARRAY_TASK_ID." >&2
    exit 1
fi

echo "#####################################################################"
echo "Ladder run $SLURM_ARRAY_TASK_ID / ${NUM_JOBS}"
echo "Config (${INDEX}): $CONFIG_FILE"
echo "Seed: $EXPERIMENT_SEED"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

cd "$SLURM_SUBMIT_DIR" || exit 1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Python puts the script's own directory on sys.path, not the working directory,
# so without this 'import src' fails.
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_MODE=online

# One evaluation protocol for all six rungs. BP restores its best-validation
# weights before testing and MF never does, so leaving that in would hand rung 1
# an accuracy advantage no other rung gets and confound every accuracy contrast.
# With this set, every rung is scored on its final weights and no rung pays for
# checkpoint I/O inside the measured wall time.
export BBP_DISABLE_CHECKPOINTS=1

srun python scripts/run_experiment.py --config "$CONFIG_FILE"
EXIT_CODE=$?

WANDB_LOCAL_DIR="${WANDB_DIR:-$SLURM_SUBMIT_DIR}/wandb"
if compgen -G "$WANDB_LOCAL_DIR/offline-run-*" > /dev/null; then
    wandb sync "$WANDB_LOCAL_DIR"/offline-run-* || echo "WARNING: wandb sync failed"
fi

deactivate

echo "#####################################################################"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Ladder run $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "Ladder run $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
echo "#####################################################################"
exit $EXIT_CODE
