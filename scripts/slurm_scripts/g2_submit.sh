#!/bin/bash
# G2 submission. ACCOUNT has no default: the author names the allocation first.
# The class grant is refused (research use unresolved, CLASS_JOB_FACT_SHEET.md).
set -euo pipefail
ACCOUNT="${ACCOUNT:?set ACCOUNT to the author-named allocation}"
[[ "$ACCOUNT" == plglscclass26* ]] && { echo "Refusing class grant $ACCOUNT"; exit 1; }
cd "$(dirname "$0")/../.." || exit 1
case "${1:-}" in
  pilot)  sbatch --account="$ACCOUNT" --array=1 --export=ALL,G2_TAG=pilot,G2_EXTRA="--measure-sec 10 --idle-sec 60" scripts/slurm_scripts/g2_sweep.slurm ;;
  sweep)  sbatch --account="$ACCOUNT" scripts/slurm_scripts/g2_sweep.slurm ;;
  bridge) sbatch --account="$ACCOUNT" --array=1-20 --export=ALL,G2_TAG=bridge,G2_EXTRA="--pipeline host --batches 128 --widths 2000" scripts/slurm_scripts/g2_sweep.slurm ;;
  init)   sbatch --account="$ACCOUNT" scripts/slurm_scripts/g2_bp_mfinit.slurm ;;
  *) echo "usage: ACCOUNT=... $0 {pilot|sweep|bridge|init}"; exit 2 ;;
esac
