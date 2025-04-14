# File: src/algorithms/__init__.py
# Import the main training/evaluation functions for each algorithm

# --- FF ---
# <<< MODIFIED: Import from the NEW ff.py implementation >>>
from .ff import (
    train_ff_model,
    evaluate_ff_model,
)
# <<< END MODIFICATION >>>

# --- CaFo ---
from .cafo import train_cafo_model, evaluate_cafo_model

# --- MF ---
from .mf import (
    mf_local_loss_fn,
    train_mf_model,
    evaluate_mf_model,
)

# --- BP Baseline ---
from src.baselines import train_bp_model, evaluate_bp_model as evaluate_bp_baseline

# --- Factory Functions ---
from typing import Callable, Dict, Any
import torch.nn as nn

def get_training_function(name: str) -> Callable:
    """Returns the main training orchestration function for an algorithm."""
    name = name.lower()
    if name == "ff":
        return train_ff_model # <<< Use the new function
    elif name == "cafo":
        return train_cafo_model
    elif name == "mf":
        return train_mf_model
    elif name == "bp":
        return train_bp_model
    else:
        raise ValueError(f"Unknown algorithm name for training: {name}")


def get_evaluation_function(name: str) -> Callable:
    """Returns the main evaluation function for an algorithm."""
    name = name.lower()
    if name == "ff":
        return evaluate_ff_model # <<< Use the new function
    elif name == "cafo":
        return evaluate_cafo_model
    elif name == "mf":
        return evaluate_mf_model
    elif name == "bp":
        return evaluate_bp_baseline
    else:
        raise ValueError(f"Unknown algorithm name for evaluation: {name}")


__all__ = [
    # FF components (Only expose main train/eval)
    "train_ff_model",
    "evaluate_ff_model",
    # CaFo components
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