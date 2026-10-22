# Cancel-and-resubmit the two pending class-account jobs

Status: **prepared only, not executed.** Both jobs have consumed **0 GPU-hours**
and are still `PENDING` on `plglscclass26-gpu-a100` with the A100
`ReqNodeNotAvail` reason. The author's 2026-10-22 account decision authorizes
new W1 work on `plgoncotherapy2-gpu-a100` (team `plggfastsim`), but the two
existing jobs are not cancelled or migrated without a separate go-ahead.

This is the proposed migration, after the author says to do it:

| job | script | tasks | action |
|---|---|---:|---|
| 3386582 | `scripts/slurm_scripts/mfglobal_tune.slurm` | 2 (50 trials/dataset) | cancel, resubmit on oncology |
| 3386586 | `scripts/slurm_scripts/w1_cap_pilot.slurm` | 5 (C100 seeds 42–46) | cancel, resubmit on oncology |

## Why

The class account has fairshare 0.000 and the A100 partition is degraded; the
jobs have made no progress. The oncology account has fairshare 0.728 and no
usage. The scripts already carry `#SBATCH -A plgoncotherapy2-gpu-a100`, so a
resubmission uses the authorized account without editing the files. No study
data or completed trials are lost: both arrays have 0 GPU-hours.

## Proposed commands (run only with explicit approval)

From `~/BeyondBackpropagation` on Athena:

```bash
# 1. Record the current state and confirm there are no completed trials.
squeue -j 3386582,3386586 -o "%.18i %.9P %.20j %.8T %.10M %.6D %R"
sacct -j 3386582,3386586 --format=JobID,JobName%24,State,Elapsed,AllocTRES%40

# 2. Cancel the two old class-account arrays.
scancel 3386582 3386586

# 3. Wait until both are gone; do not submit while the old arrays still exist.
while squeue -h -j 3386582,3386586 | grep -q .; do sleep 10; done

# 4. Resubmit under the authorized account. The scripts' #SBATCH lines match.
sbatch --account=plgoncotherapy2-gpu-a100 scripts/slurm_scripts/mfglobal_tune.slurm
sbatch --account=plgoncotherapy2-gpu-a100 scripts/slurm_scripts/w1_cap_pilot.slurm

# 5. Record the new job IDs and confirm the account.
squeue -u "$USER" -o "%.18i %.20j %.20P %.20a %.8T %.10M %R"
```

Do **not** run these commands as part of a W1 submission block. They are a
separate migration and require the author's explicit approval. The guarded
helper `scripts/slurm_scripts/resubmit_pending_oncology.sh` prints the same
commands by default and only executes when `CONFIRM=YES` is set.
