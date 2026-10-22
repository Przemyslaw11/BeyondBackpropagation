# Queue risk and W1 launch plan

Snapshot: 2026-10-22 00:20 CEST. No new job submitted for this memo; all options
below are prepared, not launched. The live jobs remain pending.

2026-10-22 recheck: both arrays are still `PENDING` with 0 GPU-hours and the same
`ReqNodeNotAvail` node list; no W1 job was submitted or cancelled.

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
- **`plgoncotherapy2-gpu-a100`** is authorized for new W1 submissions (author
  decision 2026-10-22; team `plggfastsim`). Existing jobs 3386582/3386586 remain on
  the class account; they are not cancelled or migrated.
- `8k3-gpu` / `8k3-gpu-r1` are **not** accessible with either account
  (`Access/permission denied`).
- `plgrid-now` is accessible with both accounts; it has a hard **1 h** limit.
  **Policy check:** the published Cyfronet definition (the Athena docs do not list
  a separate Athena policy) describes `plgrid-now` as *intended for interactive
  jobs*, with 1 running/queued job and 1 node maximum. A batch Optuna segment is
  not an interactive job, even with monitoring disabled. `plgrid-now` is therefore
  **not used** for validation-only tuning unless the helpdesk confirms an exemption
  in writing. Scheduler metadata (`scontrol show partition`) is not a policy.

### Scheduler estimates (GPU job, `--test-only`)

| probe | class account | oncology account |
|---|---|---|
| 4 h, 1 GPU, `plgrid-gpu-a100` | 2026-11-19 14:07 | **2026-11-12 07:20** |
| 5 h, 1 GPU, `plgrid-gpu-a100` | 2026-11-19 14:07 | **2026-11-12 07:20** |
| 1 h, 1 GPU, `plgrid-now` | **2026-10-21 22:54** | 2026-10-21 22:54 |

These are estimates from a degraded partition, not guarantees. The relative signal
is still useful: the oncology account is roughly a week earlier. The `plgrid-now`
line is a scheduler estimate only; the policy check above says it must not be used
for batch tuning. New W1 submissions use the oncology account; the two existing
pending jobs stay on the class account (no cancel/resubmit).

**Decision (2026-10-22):** new W1 submissions use `plgoncotherapy2-gpu-a100`
(team `plggfastsim`). Existing jobs 3386582/3386586 stay on
`plglscclass26-gpu-a100`; no cancellation or migration. The class account remains
the historical record for the already-submitted tuning/pilot jobs.

## 3. Segmentation plan (prepared, not submitted)

### 3.1 Tuning: 4 × 5-hour resumable segments instead of one 24 h job

Each Seq-Global trial is ~20–25 min; 50 trials is 17–21 h per dataset. The
implemented chain is `scripts/slurm_scripts/mfglobal_tune_segment.slurm` with
`submit_mfglobal_segments.sh`:

- `--target-trials 50` (new in `scripts/run_optuna_search.py`) means each
  invocation runs only `50 − stored trials` more. It is a total target, not a
  per-segment count, so a slow segment cannot overshoot 50.
- same `--study-name`; `load_if_exists=True` keeps completed trials durable.
- `--dependency=afterany:<previous>` serializes writers; CIFAR-10 and CIFAR-100
  still run as two parallel array tasks.
- `--time=05:00:00`, `--cpus-per-task=8`, `--mem=48000M`.
- Four segments cover ~20 h at 12-15 trials/segment. A killed segment loses only
  the in-flight trial; the next one resumes.
- `submit_mfglobal_segments.sh` refuses to start if an existing
  `BBP_MFGlobalTune`/`BBP_MFGlobalSeg` job is queued; cancel 3386582 before
  switching to the chain, so two writers never share the SQLite study.
- BP-sched gets the same `--target-trials 50` mechanism in
  `bp_sched_tune.slurm`, but is not chained yet.

This does not increase total GPU work (still 50 trials/study). It only replaces two
24 h queue entries with four shorter ones.

### 3.2 `plgrid-now` is not used

The 1 h partition is fast because it is an interactive/test allocation, not a
production batch queue. The published policy restricts it to interactive jobs, and
Athena's docs do not publish an exemption for validation-only tuning. A tuning
trial is still a batch job whether or not NVML is enabled. Do not use it without
written helpdesk confirmation; no `plgrid-now` chain is prepared as a fallback.

### 3.3 W1 family: frozen core first; HPO-gated arms separate

The 440-task base is six independently submittable arrays, not one array:

- `w1_family.slurm`: frozen core five arms, 200 tasks, 1 h, 8 CPU/48 GB.
- `w1_family_controls.slurm`: BP-bestval, 40 tasks, 1 h.
- `w1_bp_fixed.slurm`: BP-fixed, 40 tasks, 1 h, after the BP-sched branch;
  same frozen lr/wd, fixed 211/191 epochs, no early stopping.
- `w1_seqglobal.slurm`: Seq-Global, 40 tasks, 1 h, after the 25 Nov branch.
- `w1_controls_protocol_b.slurm`: MF-Seq and nom0 Protocol B, 80 tasks, 1 h,
  after the core-BP `Ebar_BP` freeze.
- `w1_bp_sched.slurm`: BP-sched, 40 tasks, 1 h, after the 2 Dec branch and
  placeholder replacement.

All use the same commit and seeds. 1 h is >2× the slowest observed task; a task
that needs more is a bug and should fail rather than occupy a node for 6 h. The
account switch is one line in `scripts/slurm_scripts/w1_submit.sh`; new W1
submissions default to `plgoncotherapy2-gpu-a100` (team `plggfastsim`). See
`W1_MANIFEST_ADDITIONS.md` for the 440-task count and ~118 GPU-hour estimate, and
`W1_CRITICAL_PATH.md` for the 12/19/26 Nov queue scenarios.

## 4. Pre-registered HPO fallback (approved 2026-10-22)

The author-approved rule is now in `PREREG-phase2.md` §9 and is time-triggered,
never W1-outcome-triggered:

- **Seq-Global:** 25 Nov 2026 23:59 CET.
- **BP-sched:** 2 Dec 2026 23:59 CET.
- At each deadline, freeze the best-validation trial among completed trials if
  **≥20 trials are completed per dataset**; otherwise use that arm's fallback.
- Seq-Global's fallback is MF-Seq's tuned lr/wd (`configs/mf/*`).
- BP-sched's fallback is **core BP's** tuned lr/wd (`configs/bp_baselines/*`),
  with the fixed cosine schedule; BP-fixed inherits it. It is not MF-Seq's lr/wd.
- Disclose the branch used, per dataset and per arm. The rule changes no W1
  outcome, threshold, seed, or Holm family.

`configs/mf_global/*` currently carries the MF-Seq values; `configs/bp_sched/*`
currently carries the core-BP fallback values, still marked `PLACEHOLDER` so the
launchers refuse to submit before the branch is committed. The 12/19
Nov starts can still run Seq-Global tuning before its 25 Nov deadline; the 26 Nov
start sees that deadline passed. All three starts can run BP-sched tuning before
2 Dec. The schedule assumes fallback only if the 20-trial bar is not met by the
deadline.

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
| 2026-10-22 | Account decision: new W1 submissions use `plgoncotherapy2-gpu-a100` (team `plggfastsim`). Existing jobs stay on the class account; no migration. |
| 2026-10-24 | Review whether tuning has started; if not, keep the 5 h segment chain ready. Do not use `plgrid-now`. |
| 2026-11-05 | If W1 core has not started, escalate node outage to Cyfronet; arrays are already split into 1 h core/control blocks. |
| 2026-11-12 / 19 / 26 | Queue-start scenarios; see `W1_CRITICAL_PATH.md`. |
| 2026-11-15 | DSS figure data (already generated from G1 + PPAM; see `jpdc_data/dss/`) |
| **2026-11-25 23:59 CET** | **Seq-Global HPO deadline**; record tuned/fallback branch. |
| **2026-12-02 23:59 CET** | **BP-sched HPO deadline**; record tuned/fallback branch; then BP-sched and BP-fixed. |
| 2026-12-20 | G2 scope gate |

## 7. Risks

1. **Node outage dominates.** No scheduling trick fixes 44/48 unavailable nodes.
   Escalate to Cyfronet support; keep `--test-only` probes as evidence.
2. **Account scope resolved.** `plgoncotherapy2-gpu-a100` is authorized for new W1
   submissions; pending jobs remain on the class account. Do not migrate them.
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

1. Do not submit W1 until the author says so. If authorized, submit
   `w1_submit.sh core` (200 tasks) first, then the frozen BP-bestval controls;
   Seq-Global and BP-sched/BP-fixed wait for their §9 branch commits.
2. Record the Seq-Global branch by 25 Nov and the BP-sched branch by 2 Dec.
3. Keep the 4×5 h segment chain ready but do not launch `plgrid-now`. Re-probe
   `sinfo`/`--test-only` before any resubmission; do not migrate 3386582/3386586.
