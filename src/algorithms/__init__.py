# src/algorithms/__init__.py
"""A package containing algorithm implementations and factory functions."""

from typing import Callable

from .cafo import evaluate_cafo_model, train_cafo_model
from .ff import (
    evaluate_ff_model,
    train_ff_model,
)
from .mf import (
    evaluate_mf_model,
    mf_local_loss_fn,
    train_mf_model,
)


def get_training_function(name: str) -> Callable:
    """Returns the main training orchestration function for an algorithm."""
    name = name.lower()
    if name == "ff":
        return train_ff_model
    elif name == "cafo":
        return train_cafo_model
    elif name == "mf":
        return train_mf_model
    else:
        raise ValueError(
            f"Unknown algorithm name for training: {name}. The legacy BP baseline "
            "trainers were retired; use beyond_backprop.algorithms.BPAdapter."
        )


def get_evaluation_function(name: str) -> Callable:
    """Returns the main evaluation function for an algorithm."""
    name = name.lower()
    if name == "ff":
        return evaluate_ff_model
    elif name == "cafo":
        return evaluate_cafo_model
    elif name == "mf":
        return evaluate_mf_model
    else:
        raise ValueError(
            f"Unknown algorithm name for evaluation: {name}. The legacy BP baseline "
            "evaluator was retired; use beyond_backprop.algorithms.BPAdapter."
        )


__all__ = [
    "evaluate_cafo_model",
    "evaluate_ff_model",
    "evaluate_mf_model",
    "get_evaluation_function",
    "get_training_function",
    "mf_local_loss_fn",
    "train_cafo_model",
    "train_ff_model",
    "train_mf_model",
]
