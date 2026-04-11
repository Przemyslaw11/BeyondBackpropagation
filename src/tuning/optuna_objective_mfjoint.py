"""Optuna objective for MF-Joint, rung 3 of the ablation ladder.

Shares BP-DS's trial machinery; the two differ only in which head reads out, so a
second search implementation would be a second place for them to drift apart.
"""

from typing import Any, Dict

import optuna

from src.baselines.bp_ds import evaluate_mf_joint_model, train_mf_joint_model
from src.tuning.optuna_objective_bpds import objective_deep_supervised


def objective_mfjoint(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Rung 3: MF's objective and readout, trained with joint gradients."""
    return objective_deep_supervised(
        trial, base_config, train_mf_joint_model, evaluate_mf_joint_model, "MF-Joint"
    )
