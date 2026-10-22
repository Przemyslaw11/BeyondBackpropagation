#!/bin/bash
# One account switch for all W1 arrays. Change ACCOUNT on this line only.
#
# Author decision 2026-10-22: new W1 submissions use the plgoncotherapy2 grant
# (Slurm account plgoncotherapy2-gpu-a100, team plggfastsim). Existing pending
# jobs 3386582/3386586 remain on plglscclass26-gpu-a100; do not cancel or migrate
# them. Override ACCOUNT=plglscclass26-gpu-a100 only if the author reverses this.
set -euo pipefail
ACCOUNT="${ACCOUNT:-plgoncotherapy2-gpu-a100}"
TEAM="${TEAM:-plggfastsim}"

cd "$(dirname "$0")/../.." || exit 1

echo "Submitting with account=${ACCOUNT} team=${TEAM}"

case "${1:-}" in
  core)
    # Frozen-parameter core: BP, MF-Joint, MF-Simul, MF-Seq, MF-Seq-nom0.
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_family.slurm
    ;;
  seqglobal)
    # Gated on the 2026-11-25 HPO deadline/fallback.
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_seqglobal.slurm
    ;;
  controls)
    # Frozen BP-bestval control only, independent of HPO (40 tasks).
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_family_controls.slurm
    ;;
  fixed)
    # H-W1d BP-fixed; shares the frozen BP-sched lr/wd (40 tasks).
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_bp_fixed.slurm
    ;;
  protocolb)
    # Gated on the actual core-BP Ebar_BP and a frozen budget commit.
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_controls_protocol_b.slurm
    ;;
  sched)
    # Gated on the 2026-12-02 HPO deadline/fallback and placeholder removal.
    sbatch --account="$ACCOUNT" scripts/slurm_scripts/w1_bp_sched.slurm
    ;;
  *)
    echo "Usage: $0 {core|seqglobal|controls|fixed|protocolb|sched}" >&2
    echo "ACCOUNT currently: $ACCOUNT (team $TEAM)" >&2
    exit 2
    ;;
esac
