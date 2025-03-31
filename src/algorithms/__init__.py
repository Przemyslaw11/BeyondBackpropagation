# File: src/algorithms/__init__.py
# Import the main training/evaluation functions for each algorithm

# --- FF ---
from .ff import (
    ff_loss_fn,
    train_ff_model,
    evaluate_ff_model,
)

# --- CaFo (Corrected for Rand-CE) ---
from .cafo import train_cafo_model, evaluate_cafo_model

# --- MF (Corrected) ---
from .mf import (
    mf_local_loss_fn,
    train_mf_model,
    evaluate_mf_model,
)

# Optional: Factory functions can be added here or in a separate factory module
from typing import Callable, Dict, Any
import torch.nn as nn
from src.baselines import train_bp_model, evaluate_bp_model as evaluate_bp_baseline


def get_training_function(
    name: str,
) -> Callable:  # Type hint adjusted - FF signature changed
    """Returns the main training orchestration function for an algorithm."""
    # Original FF Signature: [nn.Module, Any, Dict[str, Any], Any, Optional[Any], Optional[Callable]], None
    # New FF Signature: [nn.Module, Any, Dict[str, Any], Any, Optional[Any]], None
    # Other signatures might also differ, using generic Callable for now
    name = name.lower()
    if name == "ff":
        return train_ff_model
    elif name == "cafo":
        return (
            train_cafo_model  # Assumes train_cafo_model handles predictor creation now
        )
    elif name == "mf":
        return train_mf_model
    elif name == "bp":
        return train_bp_model  # Return the BP baseline trainer
    else:
        raise ValueError(f"Unknown algorithm name for training: {name}")


def get_evaluation_function(
    name: str,
) -> Callable:  # Type hint adjusted - FF signature changed
    """Returns the main evaluation function for an algorithm."""
    # Original FF Signature: [nn.Module, Any, Any, Optional[Callable]], Dict[str, float]
    # New FF Signature: [nn.Module, Any, Any], Dict[str, float]
    # Using generic Callable for robustness
    name = name.lower()
    if name == "ff":
        return evaluate_ff_model
    elif name == "cafo":
        # CaFo needs predictors passed potentially, evaluation function might need wrapper or config
        return evaluate_cafo_model
    elif name == "mf":
        return evaluate_mf_model
    elif name == "bp":
        # BP evaluation needs criterion
        # We need a wrapper or assume criterion is passed via config in engine
        return evaluate_bp_baseline  # Return the BP baseline evaluator
    else:
        raise ValueError(f"Unknown algorithm name for evaluation: {name}")


__all__ = [
    # FF components
    "ff_loss_fn",
    "train_ff_model",
    "evaluate_ff_model",
    # CaFo components
    # "train_cafo_predictor_only", # Keep internal unless needed externally
    "train_cafo_model",
    "evaluate_cafo_model",
    # MF components
    "mf_local_loss_fn",
    "train_mf_model",
    "evaluate_mf_model",
    # Factories
    "get_training_function",
    "get_evaluation_function",
]
