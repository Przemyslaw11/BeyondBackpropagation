"""Toy resume test for the BP-sched Optuna bookkeeping.

A segment is killed with SIGKILL while its trial is RUNNING.  The resume helper
must fail that stale trial, count only COMPLETE trials, and leave the study one
trial short of the target.  A second pass with the target reached must report
zero remaining.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

REPO = Path(__file__).resolve().parents[1]
RESUME = REPO / "scripts" / "jpdc" / "optuna_resume.py"


def _resume(db: Path, study: str, target: int) -> tuple[int, int, int]:
    completed = subprocess.run(
        [
            sys.executable,
            str(RESUME),
            "--db",
            str(db),
            "--study",
            study,
            "--target",
            str(target),
            "--shell",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(
        line.split("=", 1) for line in completed.stdout.strip().splitlines()
    )
    return (
        int(values["COMPLETE"]),
        int(values["REMAINING"]),
        int(values["FAILED_STALE"]),
    )


def _run_trials(db: Path, study: str, n: int) -> None:
    current = optuna.create_study(
        study_name=study,
        storage=f"sqlite:///{db}",
        direction="maximize",
        load_if_exists=True,
    )
    current.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=n)


def test_killed_trial_is_failed_and_only_complete_trials_count(tmp_path):
    db = tmp_path / "toy.db"
    _run_trials(db, "toy", 2)

    code = f"""
import time
import optuna

study = optuna.create_study(
    study_name="toy",
    storage="sqlite:///{db}",
    direction="maximize",
    load_if_exists=True,
)

def objective(trial):
    trial.suggest_float("x", 0.0, 1.0)
    time.sleep(30)
    return 0.0

study.optimize(objective, n_trials=1)
"""
    process = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            study = optuna.load_study(
                study_name="toy", storage=f"sqlite:///{db}"
            )
            if study.get_trials(
                deepcopy=False, states=(TrialState.RUNNING,)
            ):
                break
            time.sleep(0.1)
        else:
            pytest.fail("the toy trial never entered RUNNING")
    finally:
        process.kill()
        process.wait(timeout=10)

    assert _resume(db, "toy", 3) == (2, 1, 1)

    study = optuna.load_study(study_name="toy", storage=f"sqlite:///{db}")
    states = [trial.state for trial in study.trials]
    assert states.count(TrialState.FAIL) == 1
    assert states.count(TrialState.COMPLETE) == 2
    assert states.count(TrialState.RUNNING) == 0

    _run_trials(db, "toy", 1)
    assert _resume(db, "toy", 3) == (3, 0, 0)
