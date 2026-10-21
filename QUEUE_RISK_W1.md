# Queue risk and W1 launch plan

Snapshot: 2026-10-21 21:42 CEST. No job submitted for this memo; all options below
are proposals.

## 1. Current jobs and charge

| job | what | wall | account | state |
|---|---|---|---|---|
| **3386582** (array 1–2) | Seq-Global tuning, 50 trials per dataset | 24 h | `plglscclass26-gpu-a100` | PENDING, `ReqNodeNotAvail` |
| **3386586** (array 1–5) | H-W1cap step-1 pilot, C100, seeds 42–46 | 8 h | `plglscclass26-gpu-a100` | PENDING, `ReqNodeNotAvail` |

Both were submitted at 19:30/19:32 and have consumed **0 GPU-hours**. The pending
reason is not ordinary fair-share waiting:

```
ReqNodeNotAvail,_UnavailableNodes:t[0001-0031,0034-0036,0038-0047]
```

That is **44 of 48 A100 nodes unavailable** (failed/drained/reserved), not a queue
of runnable jobs. `sinfo` reports 23 `fail*`, 3 `drain*`, 2 `drng@`, 1 `resv`, and
only a handful of `mix`/`alloc` nodes.

## 2. Accounts available

`sshare -U` on the login node:

| account | RawUsage | EffectvUsage | FairShare | associations |
|---|---|---|---|---|
| `plglscclass26-gpu-a100` | 4857 | 0.00038 | **0.000000** | normal |
| `plgoncotherapy2-gpu-a100` | 0 | 0.00009 | **0.727783** | normal |

- Current jobs are charged to **`plglscclass26-gpu-a100`**.
- A second allocation, **`plgoncotherapy2-gpu-a100`**, is active and unused. It is a
  different project account (oncology); using it for BBP requires the grant owner's
  approval. It is not a technical question — `sbatch --test-only` accepts it.
- `8k3-gpu` / `8k3-gpu-r1` are **not** accessible with either account
  (`Access/permission denied`).
- `plgrid-now` is accessible with both accounts; it has a hard **1 h** limit.

### Scheduler estimates (GPU job, `--test-only`)

| probe | class account | oncology account |
|---|---|---|
| 4 h, 1 GPU, `plgrid-gpu-a100` | 2026-11-19 14:07 | **2026-11-12 07:20** |
| 5 h, 1 GPU, `plgrid-gpu-a100` | 2026-11-19 14:07 | **2026-11-12 07:20** |
| 1 h, 1 GPU, `plgrid-now` | **2026-10-21 22:54** | 2026-10-21 22:54 |

These are estimates from a degraded partition, not guarantees. The relative signal
is still useful: the oncology account is roughly a week earlier, and `plgrid-now`
can run a 1 h segment tonight. Note that moving accounts means cancelling and
resubmitting (age/priority is only ~2 h, so the loss is small).

**Recommendation:** ask the grant owner today whether BBP may be charged to
`plgoncotherapy2-gpu-a100`. If yes, resubmit the two pending arrays there and keep
the class account for the paper's accounting statement. If no, stay on the class
account; the root cause is the node outage, not the account.

## 3. Segmentation plan (proposal, not submitted)

### 3.1 Tuning: 5 × 10-trial segments instead of one 50-trial job

Each Seq-Global trial is ~20–25 min. A 50-trial task is 17–21 h; if the node dies at
hour 16, all uncommitted trials are lost (Optuna keeps completed ones, but the job
does not resume itself). Replace each 24 h array task with five chained 5 h jobs:

- `--n-trials 10` per segment (the flag exists in `scripts/run_optuna_search.py`).
- same `--study-name`; `load_if_exists=True` already in the script, so each segment
  adds 10 trials to the same SQLite study.
- `--dependency=afterany:<previous>` so only one process writes the SQLite file at
  a time. Two studies (CIFAR-10/CIFAR-100) still run in parallel.
- `--time=05:00:00`, `--cpus-per-task=8`, `--mem=48000M` (down from 16 CPU/128 GB).
  The 3×2000 model and 4 workers do not need 128 GB; smaller requests backfill more
  easily.
- If a segment is killed mid-trial, Optuna marks that trial failed and the next
  segment continues; record failed-trial counts in §9.

This does not increase total GPU work (still 50 trials/study), only the number of
queue entries. Ten 5 h jobs are more likely to backfill than two 24 h jobs.

### 3.2 Opportunistic `plgrid-now` segments

If the main queue has still not started by **25 Oct**, run 1 h / 2-trial segments on
`plgrid-now` (same study names, same dependencies, `--n-trials 2`, `--time=01:00:00`).
This is opportunistic only: 1 h hard cap, ~10 min startup, so 2 trials is the safe
limit. Do not count these as the full 50-trial study; they are a hedge that can give
a usable best-so-far if the main queue stays dead.

### 3.3 W1 family: split by arm, shorter wall

`w1_family.slurm` is 280 tasks × 6 h, 16 CPU/128 GB. Actual per-task times are
~5 min (BP) to ~20 min (MF-Seq). Proposal for the frozen submission:

- seven arrays of 40 tasks (BP, MF-Joint, MF-Simul, MF-Seq, Seq-Global, nom0,
  BP-long), same commit, same seeds.
- `--time=01:00:00`, `--cpus-per-task=8`, `--mem=48000M`.
- 1 h is >2× the slowest observed task; if any task needs more, it is a bug and
  should fail rather than silently occupy a node for 6 h.
- Splitting by arm lets short BP tasks backfill and gives partial arm-level progress;
  the freeze is still one commit, and `analyze_w1.py` is unchanged.

## 4. Pre-declared HPO fallback (proposed; not yet in §9)

The trigger is **time only** and never looks at W1 outcomes. Operational definition
of "not started": **no trial has completed** in the study (a started-but-killed trial
does not count).

> **Queue contingency (2026-10-21, time-triggered).** If by **2026-10-28 12:00
> Europe/Warsaw** neither Seq-Global tuning study has completed at least one trial,
> cancel job 3386582 and freeze both `configs/mf_global/*` to the inherited MF-Seq
> lr/wd: CIFAR-10 `lr=2.1536491821693216e-4`, `wd=1.606438244272842e-6`; CIFAR-100
> `lr=2.7089684419387227e-4`, `wd=7.204725563819985e-5`. If a study has completed
> 1–19 trials, use the best completed trial only if at least 10 trials are complete;
> otherwise use the inherited values. If ≥20 trials are complete, freeze the
> best-so-far at the deadline rather than wait for 50. Record the study name, trial
> number, objective, best params and commit hash in `PREREG-phase2.md` §9, then
> submit the W1 family at that commit. This rule uses wall-clock only and does not
> depend on any W1 result.

Why 10 trials: with 50-trial TPE on a 2-D lr/wd space, 10 trials is a weak but
non-degenerate search; below 10 the "best" is mostly sampler noise. The inherited
values are the published MF-Seq tuned values, so the fallback is still a defensible
Seq-Global arm (it differs from MF-Seq only in the schedule).

**Status:** this wording is a proposal for approval. It must be inserted into §9
before 28 Oct to be genuinely pre-registered; no PREREG file has been edited for it.

## 5. Parallel workers? No.

Two studies already run in parallel (one per dataset). More workers would:

- contend on the same SQLite file (write locks; Optuna retries, but trials can fail);
- compete for the same degraded A100 partition, where there are almost no free nodes;
- not shorten the critical path while `ReqNodeNotAvail` is the reason.

If SQLite contention becomes real, the safe fix is one writer per study (as above),
not a second worker on the same study.

## 6. Decision points

| date | decision |
|---|---|
| 2026-10-22 | Ask grant owner about charging BBP to `plgoncotherapy2-gpu-a100`; decide account |
| 2026-10-24 | Review whether tuning has started; if not, prepare `plgrid-now` 2-trial chain |
| **2026-10-28 12:00** | **HPO fallback trigger**; freeze Seq-Global and submit W1 family |
| 2026-11-05 | If W1 family has not started, split arrays / escalate node outage to Cyfronet |
| 2026-11-15 | DSS figure data (already generated from G1 + PPAM; see `jpdc_data/dss/`) |
| 2026-12-20 | G2 scope gate |

## 7. Risks

1. **Node outage dominates.** No scheduling trick fixes 44/48 unavailable nodes.
   Escalate to Cyfronet support; keep `--test-only` probes as evidence.
2. **Account scope.** `plgoncotherapy2-gpu-a100` is technically valid but belongs to
   another project. Do not switch without written approval.
3. **SQLite on kill.** Chained segments reduce exposure; Optuna completed trials are
   durable. Back up the `.db` after each segment (`cp` to a timestamped file).
4. **W1 overwrites G1 directories.** G1 archive already exists at
   `results/archive_g1_20261021/` and locally in `jpdc_data/g1/pull/`; verify before
   submission.
5. **Fallback looks like a result.** The fallback is time-triggered and fixed before
   any W1 outcome; record the trigger date and trial counts in §9, and report the
   Seq-Global arm as "inherited MF-Seq hyperparameters" if it fires.
6. **Quota.** Tuning stores SQLite + logs in `results/optuna`; HOME is 10 GiB. Keep
   `WANDB_DIR=$SCRATCH/bbp_wandb`, `BBP_DISABLE_CHECKPOINTS=1`, and prune old logs.

## 8. Next three actions

1. Get the account-scope answer; if yes, cancel/resubmit 3386582/3386586 to
   `plgoncotherapy2-gpu-a100` (test-only estimate 12 Nov vs 19 Nov).
2. Approve (or edit) the §9 fallback wording above before 28 Oct.
3. If tuning is still pending on 25 Oct, launch the `plgrid-now` 2-trial chain as a
   hedge; otherwise keep the 5×10 segment plan ready for the moment nodes return.
