#!/usr/bin/env bash
#SBATCH --job-name=BBP_Array_%a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
# The submitter supplies account/partition and may override resources with sbatch.

set -u
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$ROOT" || exit 1
PYTHON_BIN=${PYTHON_BIN:-python3}
SRUN_BIN=${SRUN_BIN:-srun}
CONFIG_DIR=${CONFIG_DIR:-configs}
# The manifest maps 1-based array task IDs to configuration files.  A manifest
# provided through CONFIG_MANIFEST is trusted as-is.  Otherwise it is generated
# exactly once per array job and published with an atomic rename so that
# concurrent tasks never observe a truncated or partially written file.
if [[ -n "${CONFIG_MANIFEST:-}" ]]; then
    MANIFEST=$CONFIG_MANIFEST
    if [[ ! -s "$MANIFEST" ]]; then
        echo "Error: CONFIG_MANIFEST does not exist or is empty: $MANIFEST" >&2
        exit 2
    fi
else
    MANIFEST=$ROOT/results/config-manifest-${SLURM_ARRAY_JOB_ID:-standalone}.txt
    mkdir -p "$(dirname "$MANIFEST")"
    if [[ ! -s "$MANIFEST" ]]; then
        GENERATE_MANIFEST=1
        if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
            # Array mode: let exactly one task generate the manifest while the
            # remaining tasks wait for its atomic publication.
            LOCK=$ROOT/results/.config-manifest-$SLURM_ARRAY_JOB_ID.lock
            if mkdir "$LOCK" 2>/dev/null; then
                trap 'rm -rf "$LOCK"' EXIT
            else
                GENERATE_MANIFEST=0
                for _ in $(seq 1 120); do
                    [[ -s "$MANIFEST" ]] && break
                    sleep 2
                done
                if [[ ! -s "$MANIFEST" ]]; then
                    echo "Error: timed out waiting for manifest $MANIFEST; the generating task may have failed (lock: $LOCK)." >&2
                    exit 2
                fi
            fi
        fi
        if [[ "$GENERATE_MANIFEST" -eq 1 ]]; then
            TMP_MANIFEST=$(mktemp "$(dirname "$MANIFEST")/.config-manifest.XXXXXX")
            "$PYTHON_BIN" - "$CONFIG_DIR" "$TMP_MANIFEST" <<'PY'
from pathlib import Path
import sys

from beyond_backprop.config import load_experiment_config

config_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
paths = sorted(path for path in config_dir.rglob("*.yaml") if path.name != "base.yaml")
if not paths:
    raise SystemExit(f"No YAML configurations found under {config_dir}")
for path in paths:
    load_experiment_config(path)
manifest.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
print(f"validated {len(paths)} configurations")
PY
            manifest_status=$?
            if [[ $manifest_status -ne 0 ]]; then
                rm -f "$TMP_MANIFEST"
                exit $manifest_status
            fi
            mv -f "$TMP_MANIFEST" "$MANIFEST"
            echo "manifest published: $MANIFEST"
        fi
    fi
fi

CONFIG_FILES=()
while IFS= read -r _manifest_line; do
    [[ -n "$_manifest_line" ]] && CONFIG_FILES+=("$_manifest_line")
done < "$MANIFEST"
NUM_CONFIGS=${#CONFIG_FILES[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 1 || TASK_ID > NUM_CONFIGS )); then
    echo "Error: invalid SLURM_ARRAY_TASK_ID '$TASK_ID' for $NUM_CONFIGS configurations." >&2
    exit 2
fi
CONFIG_FILE=${CONFIG_FILES[$((TASK_ID - 1))]}
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: manifest config not found: $CONFIG_FILE" >&2
    exit 2
fi

command=("$PYTHON_BIN" -m beyond_backprop experiment run --config "$CONFIG_FILE")
if command -v "$SRUN_BIN" >/dev/null 2>&1; then
    "$SRUN_BIN" "${command[@]}"
else
    "${command[@]}"
fi
exit $?
