# W1 manifest additions and budget estimate

Status: 2026-10-22, **not submitted**. The six core arms remain the primary
family; everything here is a control or a robustness check. No threshold, Holm
family, endpoint or seed changes. Parameters that are frozen are split from the
two HPO-gated arms so the former can be submitted independently.

## Task count

| block | arms | configs | tasks | submittable when |
|---|---|---|---:|---|
| frozen core | BP, MF-Joint, MF-Simul, MF-Seq, MF-Seq-nom0 | 10 | 200 | now, author gate only |
| frozen controls | BP-long, BP-bestval | 4 | 80 | now, author gate only |
| Seq-Global | Seq-Global | 2 | 40 | 4 Nov 2026 branch recorded/committed |
| Protocol B | MF-Seq, MF-Seq-nom0 Protocol B | 4 | 80 | core-BP `Ebar_BP` frozen/committed |
| BP-sched | BP cosine-schedule, fixed 100 epochs | 2 | 40 | 11 Nov 2026 branch recorded/committed |
| **base total** | | **22** | **440** | |
| conditional | MF-Seq cap-300 | 2 | 40 | only if H-W1cap step 1 triggers |

BP under Protocol B is the core BP arm; it is not duplicated. The earlier
280-task manifest is superseded by this 440-task base (see
`PREREG-phase2.md` §9, 2026-10-21 amendment).

## Additions, in priority order

1. **BP-sched.** `configs/bp_sched/*`; cosine `T_max=100`, no early stopping,
   fixed 100 epochs, lr/wd from a 50-trial-per-dataset study
   (`configs/tuning/bp_sched_*`); at the 11 Nov 2026 23:59 CET deadline, if
   fewer than 20 trials completed per dataset, reuse the MF-Seq lr/wd per §9.
   The BP Optuna objective now honours `training.scheduler`; otherwise tuning
   and confirmation would use different schedules. 40 confirm tasks, 0 new
   trials in the confirm run.
2. **BP-bestval.** `configs/bp_bestval/*`; same optimizer/early stopping as core
   BP, restoring the best validation-accuracy state from host memory before the
   final test. No disk checkpoint is written. 0 new tuning trials; inherits core
   BP lr/wd.
3. **MF-Seq-nom0 Protocol B.** `configs/mf_nom0_protocol_b/*`; `mf_train_m0:
   false`, `S=L=3`, per-stage budget `ceil(Ebar_BP/3)`. Planning values from G1
   `Ebar_BP = 52.95/43.90`: **18/15** epochs per stage. 40 tasks.
4. **Contemporaneous Protocol B MF-Seq.** `configs/mf_protocol_b/*`; `S=4`,
   budget `ceil(Ebar_BP/4)`: planning **14/11** epochs per stage. 40 tasks. BP
   Protocol B is the core BP arm (0 new tasks).

Budget freeze rule: after the 20 core BP runs finish, compute
`Ebar_BP = mean(passes)` per dataset, set
`budget = ceil(Ebar_BP / S)`, write it into the two Protocol-B configs, record
the values and the commit in `PREREG-phase2.md` §9, and only then launch the
controls. G1 planning values are not results and are not used for inference.

## GPU-hour estimate

Durations are measured G1 means; Seq-Global and BP-sched are conservative
estimates. 20 seeds per dataset.

| block | C10 s/run | C100 s/run | GPU-h |
|---|---:|---:|---:|
| core BP | 265 | 220 | 2.7 |
| core MF-Joint | 258 | 250 | 2.8 |
| core MF-Simul | 288 | 179 | 2.6 |
| core MF-Seq | 1043 | 950 | 11.1 |
| core MF-Seq-nom0 (Seq minus M0) | 804 | 736 | 8.6 |
| **frozen core subtotal (5 arms)** | | | **27.7** |
| core Seq-Global (est. 1300/1200) | 1300 | 1200 | 13.9 |
| **core total (6 arms)** | | | **41.6** |
| BP-long | 265 | 220 | 2.7 |
| BP-bestval | 265 | 220 | 2.7 |
| MF-Seq Protocol B (4×14/11 epochs) | 276 | 219 | 2.7 |
| nom0 Protocol B (3×18/15 epochs) | 267 | 224 | 2.7 |
| **controls total** | | | **10.9** |
| BP-sched confirm (100 epochs) | 501 | 500 | 5.6 |
| **confirm total (440 tasks)** | | | **58.0** |
| Seq-Global tuning (100 trials, 50/dataset) | | | 37.5 |
| BP-sched tuning (100 trials, 50/dataset) | | | 13.9 |
| **tuning total (if both studies run)** | | | **51.4** |
| **overall (if both studies run)** | | | **~109** |

The already-submitted Seq-Global job (3386582) is the 37.5 GPU-h tuning block,
if it ever starts. The remaining new work is ~72 GPU-h. These are compute-hours,
not wall time; the 1 h arrays can backfill, the tuning needs 5 h segments. If a
deadline fallback triggers, the corresponding tuning block is not spent; the 440
confirm tasks are unchanged.

BP-long is costed at the observed BP stopping point (~53/44 passes). If its
patience never fires and it runs to the 300-epoch cap, that block rises from
2.7 to ~16.7 GPU-h; the overall estimate is then ~123 GPU-h. The cap-300
conditional arm is not included in either number.

## Submission order and account

1. Frozen core: `scripts/slurm_scripts/w1_submit.sh core` (200 tasks, 1 h).
2. Frozen controls: `w1_submit.sh controls` (80 tasks, 1 h), independently of
   Seq-Global and BP-sched tuning.
3. After core BP finishes: compute `Ebar_BP`, freeze the Protocol-B integers,
   commit, then `w1_submit.sh protocolb` (80 tasks, 1 h).
4. Seq-Global: record the 4 Nov branch (tuned trial or MF-Seq fallback), commit,
   then `w1_submit.sh seqglobal` (40 tasks, 1 h).
5. BP-sched: record the 11 Nov branch, replace `PLACEHOLDER` in
   `configs/bp_sched/*` with the chosen lr/wd, commit, then `w1_submit.sh sched`
   (40 tasks, 1 h).

`w1_submit.sh` holds the single account line. New W1 submissions use
`plgoncotherapy2-gpu-a100` (team `plggfastsim`), as authorized on 2026-10-22.
Existing pending jobs 3386582/3386586 remain on `plglscclass26-gpu-a100` and are
not migrated. No W1 job has been submitted.

`plgrid-now` is not used. The published Cyfronet definition says it is for
interactive jobs (1 h, 1 running/queued job, 1 node); a 5 h batch tuning segment
is not an interactive job. Athena's current documentation does not publish a
separate exemption.
