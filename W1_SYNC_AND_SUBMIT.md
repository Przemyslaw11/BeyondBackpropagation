# W1 sync and submission draft — DO NOT RUN YET

Status: **draft, 2026-10-22. Nothing here has been executed.** The author has not
approved a W1 submission, and the cluster checkout is still at `813430e`. This
document is the copy-paste procedure for when that approval arrives. It does not
submit anything by itself.

The frozen local branch is `jpdc-extension`; the cluster working tree is
`athena:~/BeyondBackpropagation`. All new W1 work uses
`plgoncotherapy2-gpu-a100` (team `plggfastsim`). Existing pending class-account
jobs 3386582/3386586 are **not** touched here; their separate migration draft is
`RESUBMIT_PENDING_JOBS.md`.

## 0. Local preflight (no cluster action)

Run from the repo root. The tree must be clean; the commit below is the freeze
commit that will be recorded in `PREREG-phase2.md` §9.

```bash
cd ~/Desktop/BeyondBackpropagation_ready/BeyondBackpropagation
git status --short --branch
git log --oneline -3
PYTHONPATH=. python3 -m pytest -q
bash -n scripts/slurm_scripts/w1_family.slurm
bash -n scripts/slurm_scripts/w1_family_controls.slurm
bash -n scripts/slurm_scripts/w1_bp_fixed.slurm
bash -n scripts/slurm_scripts/w1_controls_protocol_b.slurm
bash -n scripts/slurm_scripts/w1_seqglobal.slurm
bash -n scripts/slurm_scripts/w1_bp_sched.slurm
FROZEN_COMMIT=$(git rev-parse HEAD)
echo "FROZEN_COMMIT=$FROZEN_COMMIT"
```

If the tree is not clean, stop. Do not bundle an uncommitted tree.

## 1. Bundle and copy to Athena (one block)

The cluster already has the base commit `813430e`; the bundle carries only the
freeze history on top of it.

```bash
cd ~/Desktop/BeyondBackpropagation_ready/BeyondBackpropagation
set -euo pipefail
FROZEN_COMMIT=$(git rev-parse HEAD)
BUNDLE="/tmp/bbp-${FROZEN_COMMIT}.bundle"
git bundle create "$BUNDLE" "813430e..refs/heads/jpdc-extension"
git bundle verify "$BUNDLE"
scp "$BUNDLE" athena:/tmp/
echo "copied $BUNDLE to athena:/tmp/"
```

## 2. Fetch and fast-forward the cluster checkout (one block)

This aborts if the cluster tree is dirty, if `jpdc-extension` has diverged, or if
the checked-out commit does not equal the local freeze commit. It does not run
any training.

```bash
cd ~/Desktop/BeyondBackpropagation_ready/BeyondBackpropagation
set -euo pipefail
FROZEN_COMMIT=$(git rev-parse HEAD)
BUNDLE="/tmp/bbp-${FROZEN_COMMIT}.bundle"
ssh athena "FROZEN_COMMIT='${FROZEN_COMMIT}' BUNDLE='${BUNDLE}' bash -s" <<'REMOTE'
set -euo pipefail
cd "$HOME/BeyondBackpropagation"
if [ -n "$(git status --porcelain)" ]; then
    echo "cluster tree is dirty; refusing to overwrite:" >&2
    git status --short >&2
    exit 1
fi
git fetch "$BUNDLE" refs/heads/jpdc-extension
git checkout jpdc-extension
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = "$FROZEN_COMMIT" || {
    echo "HEAD $(git rev-parse HEAD) != $FROZEN_COMMIT" >&2
    exit 1
}
git status --short --branch
git log --oneline -3
REMOTE
```

If the cluster has local commits on `jpdc-extension`, the fast-forward fails
safely; inspect before doing anything else.

## 3. Cluster preflight (read-only)

```bash
ssh athena 'cd "$HOME/BeyondBackpropagation" && git rev-parse HEAD && git status --short --branch && squeue -u "$USER" -o "%.18i %.20j %.20P %.20a %.8T %.10M %R" && sinfo -p plgrid-gpu-a100 -o "%.12P %.6a %.10l %.6D %.6t %N"'
```

Record the printed commit. It must equal `FROZEN_COMMIT`. Do not submit if
`squeue` shows an existing W1 array on the same experiments.

## 4. Submission — one command per block

These are the only W1 submission commands. Do not submit the whole 440-task
base at once. Run the blocks in this order.

### Block A — frozen core (200 tasks)

```bash
ssh athena 'cd "$HOME/BeyondBackpropagation" && ./scripts/slurm_scripts/w1_submit.sh core'
```

### Block B — frozen BP-bestval control (40 tasks)

```bash
ssh athena 'cd "$HOME/BeyondBackpropagation" && ./scripts/slurm_scripts/w1_submit.sh controls'
```

### Block C — Protocol B (80 tasks; ONLY after core-BP `Ebar_BP` is frozen and committed)

The gate is a commit that writes the actual `ceil(Ebar_BP/S)` integers into
`configs/mf_protocol_b/*` and `configs/mf_nom0_protocol_b/*` and records them in
`PREREG-phase2.md` §9. Do not use the G1 planning values as if they were results.

```bash
ssh athena 'cd "$HOME/BeyondBackpropagation" && ./scripts/slurm_scripts/w1_submit.sh protocolb'
```

### Block D — Seq-Global (40 tasks; ONLY after the 25 Nov branch is recorded)

```bash
ssh athena 'cd "$HOME/BeyondBackpropagation" && ./scripts/slurm_scripts/w1_submit.sh seqglobal'
```

### Block E — BP-sched and BP-fixed (40 + 40 tasks; ONLY after the 2 Dec branch is recorded)

Both are gated on the same frozen lr/wd in `configs/bp_sched/*`; `w1_bp_fixed.slurm`
refuses to run while `PLACEHOLDER` remains.

```bash
ssh athena 'cd "$HOME/BeyondBackpropagation" && test -z "$(grep -R PLACEHOLDER configs/bp_sched || true)" && ./scripts/slurm_scripts/w1_submit.sh sched && ./scripts/slurm_scripts/w1_submit.sh fixed'
```

If the `grep` finds a placeholder, the `&&` chain stops before submitting. Do not
replace the placeholders with core-BP values by hand; use the recorded HPO branch
or the MF-Seq fallback and commit it first.

## 5. Post-submission checks

```bash
ssh athena 'squeue -u "$USER" -o "%.18i %.20j %.20P %.20a %.8T %.10M %R" && sacct -u "$USER" --starttime=today --format=JobID,JobName%24,Account,Partition,State,Elapsed,AllocTRES%40'
```

Record the new array IDs in `PREREG-phase2.md` §9 and in the session report. Verify
that every W1 array is on `plgoncotherapy2-gpu-a100`, not the class account.

## 6. Abort / rollback

- To stop a block: `scancel <array_job_id>`; do not cancel core runs that have
  already produced complete seeds.
- Do not migrate 3386582/3386586 as part of a W1 rollback.
- If the cluster commit differs from `FROZEN_COMMIT`, stop and re-sync; never
  train on a mixed checkout.
- W3/W4 are CPU work and are never blocked by a W1 queue rollback.
