# W1 critical path — queue starts 12, 19, 26 Nov 2026

Status: 2026-10-22, **no W1 job submitted**. This document freezes the order of
operations, not a submission. The author must still say "submit".

Account decision: new W1 submissions use `plgoncotherapy2-gpu-a100`
(team `plggfastsim`). Existing jobs 3386582 and 3386586 stay on
`plglscclass26-gpu-a100`; they are neither cancelled nor migrated.

## 1. HPO deadlines change the dependency graph

The author-approved fallback (PREREG-phase2.md §9, 2026-10-22) is:

- Seq-Global deadline **25 Nov 2026 23:59 CET**;
- BP-sched deadline **2 Dec 2026 23:59 CET**;
- freeze the best-validation trial among completed trials if **≥20 completed
  per dataset**, otherwise use the arm's fallback: Seq-Global → MF-Seq's lr/wd;
  BP-sched → **core BP's** lr/wd with the cosine schedule (BP-fixed inherits).
  Disclose the branch.

The deadlines were moved on 2026-10-22 for the A100 outage, not for data; no
tuning trial had completed. The 12 and 19 Nov starts can still run the tuning
arrays before the Seq-Global deadline if the queue cooperates; the 26 Nov start
sees the Seq-Global deadline already passed. The BP-sched deadline is 2 Dec, so
all three starts can run its tuning in parallel with the frozen core. The
baseline if tuning has not met the ≥20-per-dataset bar by the deadline is the
fallback branch. The branch must be recorded in §9 and committed before the
corresponding confirm array is submitted.

This makes the frozen core independent of both tuning studies:

| block | tasks | tuning dependency | submit condition |
|---|---:|---|---|
| Frozen core: BP, MF-Joint, MF-Simul, MF-Seq, nom0 | 200 | none | author says submit |
| Frozen controls: BP-bestval | 40 | none | author says submit |
| Seq-Global | 40 | 25 Nov branch recorded | fallback or tuned config committed |
| BP-fixed | 40 | BP-sched branch recorded | placeholders replaced and committed; same lr/wd |
| BP-sched | 40 | 2 Dec branch recorded | placeholders replaced and committed |
| Protocol B: MF-Seq, nom0 | 80 | none after core BP | actual `Ebar_BP` budgets frozen/committed |
| cap-300 conditional | 40 | none | cap pilot triggers H-W1cap step 2; outside 440 |

`w1_submit.sh` exposes these as `core`, `controls`, `fixed`, `seqglobal`,
`protocolb`, and `sched`. No mode submits the whole 440-task base at once.

## 2. Queue-start scenarios

Assume 1 h array blocks and that the scheduler grants at least a few concurrent
A100s. The estimates are ordering aids, not queue predictions.

### Start 12 Nov

1. Submit **core 200** immediately; this is the only never-cut GPU block.
2. Submit **frozen controls 40** (BP-bestval) alongside the core.
3. Let the Seq-Global tuning array run; at the 25 Nov deadline record either
   the tuned branch (≥20 completed per dataset) or the fallback, commit, then
   submit **Seq-Global 40**.
4. Let the BP-sched tuning array run; at the 2 Dec deadline record either the
   tuned branch or the fallback, replace both `PLACEHOLDER` lr/wd lines, commit,
   then submit **BP-sched 40** and **BP-fixed 40** (same lr/wd, fixed epochs).
5. Once core BP has produced its seed set, compute `Ebar_BP`, freeze the
   Protocol-B integers, commit, submit **Protocol B 80**.
6. Expected finish: late Nov with normal throughput; December is buffer for
   reruns and the G2 gate (20 Dec).

### Start 19 Nov

1. Submit **core 200** first; do not wait for Seq-Global or BP-sched.
2. Submit **frozen controls 40** if slots exist; otherwise queue them next.
3. Freeze Protocol B from the core-BP mean and submit **Protocol B 80**.
4. Seq-Global tuning can still run to its 25 Nov deadline. Record the branch and
   submit **Seq-Global 40** on 25–26 Nov.
5. BP-sched tuning can run to 2 Dec. Record the branch, submit **BP-sched 40**
   and **BP-fixed 40**; if throughput is tight, defer these to the next wave,
   never the core.

### Start 26 Nov

1. Submit **core 200** immediately.
2. Submit **Protocol B 80** after the core-BP budget freeze; submit **frozen
   controls 40** (BP-bestval) in parallel.
3. The Seq-Global deadline has passed: record the branch and submit
   **Seq-Global 40**. BP-sched can run to 2 Dec; record the branch and submit
   **BP-sched 40** plus **BP-fixed 40**.
4. The GPU work is ~440 one-hour blocks; with even modest concurrency it can
   finish before the 8 Jan 2027 experiment freeze. The binding constraint is
   queue availability, not compute.
5. If only one wave is possible, the order is core → Protocol B → controls →
   Seq-Global → BP-sched/BP-fixed. The last two may be reported with fallback HPO and a
   disclosed reduced tuning budget; they are never replaced by a new arm.

## 3. Scope-cut order under deadline pressure

The author's cut order is:

1. **W3 comms** — first to be cut from the JPDC claims. It is a zero-GPU
   accounting addendum and does not consume queue time; cutting it changes no W1
   result.
2. **W4** — second to be cut. The within-session ratio model is zero-GPU and
   descriptive; it does not gate any W1 arm.
3. **W1 core is never cut.** No version of this schedule trades a core arm for
   W3 or W4.

W3/W4 can be run on CPU while the GPU queue is blocked; doing them does not
delay W1. "Cut" above means remove from the paper's claims, not delete the
artifacts.

## 4. Hard gates

- Do not submit any W1 array before the author says so.
- Do not submit Seq-Global before the 25 Nov branch is recorded.
- Do not submit BP-sched or BP-fixed while `configs/bp_sched/*.yaml` contains
  `PLACEHOLDER`, or before the 2 Dec branch is recorded.
- Do not submit Protocol B before the core-BP `Ebar_BP` freeze is committed.
- Do not migrate the two pending class-account jobs to the oncology account.
- Sync the cluster checkout to the frozen local commit immediately before the
  first submission.
