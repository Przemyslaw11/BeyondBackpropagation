#!/usr/bin/env bash
#SBATCH --job-name=BBP_Experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Account, partition, and resource overrides are supplied by the submitter or
# the cluster profile (for example: sbatch -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION").

set -u
CONFIG_FILE=${1:-}
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: no configuration file path provided." >&2
    exit 2
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: configuration file not found: $CONFIG_FILE" >&2
    exit 2
fi

ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$ROOT" || exit 1
PYTHON_BIN=${PYTHON_BIN:-python3}
SRUN_BIN=${SRUN_BIN:-srun}

"$PYTHON_BIN" -m beyond_backprop validate-config --config "$CONFIG_FILE" || exit $?
command=("$PYTHON_BIN" -m beyond_backprop experiment run --config "$CONFIG_FILE")
if command -v "$SRUN_BIN" >/dev/null 2>&1; then
    "$SRUN_BIN" "${command[@]}"
else
    "${command[@]}"
fi
exit $?
