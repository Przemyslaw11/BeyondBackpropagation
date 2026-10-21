#!/bin/bash
# One account switch for all W1 arrays. Change ACCOUNT on this line only.
# Default stays on the class grant; do not switch without owner approval.
set -euo pipefail
ACCOUNT="${ACCOUNT:-plglscclass26-gpu-a100}"
# Alternate, only after written approval: plgoncotherapy2-gpu-a100

cd "$(dirname "$0")/../.." || exit 1

case "${1:-}" in
  core)
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_family.slurm
    ;;
  controls)
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_family_controls.slurm
    ;;
  sched)
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_bp_sched.slurm
    ;;
  *)
    echo "Usage: $0 {core|controls|sched}" >&2
    echo "ACCOUNT currently: $ACCOUNT" >&2
    exit 2
    ;;
esac
