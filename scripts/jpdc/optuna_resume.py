#!/usr/bin/env python3
"""Resume bookkeeping for the BP-sched Optuna studies.

The pre-registered budget is 50 **completed** trials per dataset.  A killed
segment can leave a trial in state RUNNING and can leave FAIL/PRUNED rows, so
the remainder must be computed from COMPLETE trials only:

* fail every stale RUNNING trial (``study.tell(..., FAIL)``);
* count ``TrialState.COMPLETE``;
* print ``COMPLETE``, ``REMAINING`` and ``FAILED_STALE``.

The script imports Optuna only (no torch, no CUDA).  It never changes the
SQLite journal mode; SQLite's default rollback journal is used, never WAL.
"""

from __future__ import annotations

import argparse
import sys

import optuna
from optuna.trial import TrialState


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="SQLite file")
    parser.add_argument("--study", required=True, help="Optuna study name")
    parser.add_argument("--target", type=int, required=True, help="budget")
    parser.add_argument(
        "--shell",
        action="store_true",
        help="print COMPLETE=/REMAINING=/FAILED_STALE= for eval",
    )
    args = parser.parse_args()

    storage = f"sqlite:///{args.db}"
    try:
        study = optuna.load_study(study_name=args.study, storage=storage)
    except KeyError:
        study = None

    failed_stale = 0
    if study is not None:
        stale = study.get_trials(
            deepcopy=False, states=(TrialState.RUNNING,)
        )
        for trial in stale:
            study.tell(trial.number, state=TrialState.FAIL)
            failed_stale += 1
        complete = sum(
            1 for trial in study.trials if trial.state == TrialState.COMPLETE
        )
    else:
        complete = 0

    remaining = max(0, args.target - complete)
    if args.shell:
        print(f"COMPLETE={complete}")
        print(f"REMAINING={remaining}")
        print(f"FAILED_STALE={failed_stale}")
    else:
        print(
            f"complete={complete} remaining={remaining} "
            f"failed_stale={failed_stale}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
