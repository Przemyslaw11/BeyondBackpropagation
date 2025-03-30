# File: src/baselines/__init__.py
from .bp import train_bp_model, evaluate_bp_model, train_bp_epoch

__all__ = [
    "train_bp_model",
    "evaluate_bp_model",
    "train_bp_epoch",  # Expose epoch function for Optuna objective
]
