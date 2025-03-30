# File: src/algorithms/__init__.py

# Import the main training/evaluation functions for each algorithm

# --- FF ---
from .ff import (
    # train_ff_layer, # Keep internal unless needed externally
    ff_loss_fn,
    train_ff_model,
    evaluate_ff_model,
)

# --- CaFo (Corrected for Rand-CE) ---
from .cafo import train_cafo_predictor_only, train_cafo_model, evaluate_cafo_model

# --- MF (Corrected) ---
from .mf import (
    # train_mf_hidden_layer, # Keep internal
    # train_mf_output_layer, # Keep internal
    train_mf_model,
    evaluate_mf_model,
    mf_local_loss_fn,
)

# Optional: Factory functions can be added here or in a separate factory module
# Example:
# from typing import Callable
# def get_training_function(name: str) -> Callable:
#     """Returns the main training orchestration function for an algorithm."""
#     name = name.lower()
#     if name == 'ff':
#         return train_ff_model
#     elif name == 'cafo':
#         return train_cafo_model # Assumes train_cafo_model handles predictor creation now
#     elif name == 'mf':
#         return train_mf_model
#     # BP might be imported from baselines
#     # elif name == 'bp':
#     #     from src.baselines import train_bp_model
#     #     return train_bp_model
#     else:
#         raise ValueError(f"Unknown algorithm name for training: {name}")

# def get_evaluation_function(name: str) -> Callable:
#     """Returns the main evaluation function for an algorithm."""
#     name = name.lower()
#     if name == 'ff':
#         return evaluate_ff_model
#     elif name == 'cafo':
#         # Consider wrapping evaluate_cafo_model if default behavior (e.g., last predictor) is needed
#         # Or configure predictor selection/aggregation via config passed to evaluate_cafo_model
#         return evaluate_cafo_model
#     elif name == 'mf':
#         return evaluate_mf_model
#     # BP might be imported from baselines
#     # elif name == 'bp':
#     #     from src.baselines import evaluate_bp_model
#     #     return evaluate_bp_model
#     else:
#         raise ValueError(f"Unknown algorithm name for evaluation: {name}")


__all__ = [
    # FF components
    "ff_loss_fn",
    "train_ff_model",
    "evaluate_ff_model",
    # CaFo components
    "train_cafo_predictor_only",  # Expose if needed for variants
    "train_cafo_model",
    "evaluate_cafo_model",
    # MF components
    "mf_local_loss_fn",
    "train_mf_model",
    "evaluate_mf_model",
    # Factories (Optional)
    # "get_training_function",
    # "get_evaluation_function",
]
