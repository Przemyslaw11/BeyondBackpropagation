"""Optuna objective for MF-Simul, rung 3b of the ablation ladder.

Shares BP-DS's trial machinery; unlike MF-Joint it detaches the input of every
hidden layer, so each local loss reaches only its own W_i and M_i while the
schedule stays joint.
"""

from typing import Any, Dict

import optuna

from src.baselines.bp_ds import evaluate_mf_simul_model, train_mf_simul_model
from src.tuning.optuna_objective_bpds import objective_deep_supervised


def objective_mfsimul(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Rung 3b: MF's objective and readout, joint schedule, detached inputs."""
    return objective_deep_supervised(
        trial, base_config, train_mf_simul_model, evaluate_mf_simul_model, "MF-Simul"
    )
