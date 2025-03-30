# File: src/algorithms/__init__.py

# Import the main training/evaluation functions for each algorithm

# --- FF ---
from .ff import (
    train_ff_layer,
    # generate_positive_negative_data, # Likely internal use now
    ff_loss_fn,
    train_ff_model,  # Orchestrator
    evaluate_ff_model,  # Evaluation
)

# --- CaFo ---
from .cafo import train_cafo_block_and_predictor, train_cafo_model, evaluate_cafo_model

# --- MF (Corrected) ---
from .mf import (
    train_mf_hidden_layer,
    train_mf_output_layer,
    train_mf_model,
    evaluate_mf_model,
    mf_local_loss_fn,
)

# Optional: Define a factory function or dictionary to get algorithm functions by name
# def get_training_function(name: str) -> callable:
#     name = name.lower()
#     if name == 'ff':
#         return train_ff_model # Now implemented
#     elif name == 'cafo':
#         return train_cafo_model
#     elif name == 'mf':
#         return train_mf_model
#     elif name == 'bp': # Maybe add BP here? Or keep separate
#         from src.baselines import train_bp_model
#         return train_bp_model
#     else:
#         raise ValueError(f"Unknown algorithm name for training: {name}")

# def get_evaluation_function(name: str) -> callable:
#      name = name.lower()
#      if name == 'ff':
#          return evaluate_ff_model # Now implemented
#      elif name == 'cafo':
#          # CaFo eval needs specific predictor index handling, maybe wrap it
#          # def wrapped_cafo_eval(*args, **kwargs):
#          #     return evaluate_cafo_model(*args, use_predictor_index=-1, **kwargs)
#          # return wrapped_cafo_eval
#          return evaluate_cafo_model # Or pass index via config
#      elif name == 'mf':
#          return evaluate_mf_model
#      elif name == 'bp':
#          from src.baselines import evaluate_bp_model
#          return evaluate_bp_model
#      else:
#          raise ValueError(f"Unknown algorithm name for evaluation: {name}")


__all__ = [
    # FF components
    "train_ff_layer",
    # "generate_positive_negative_data", # Comment out if internal
    "ff_loss_fn",
    "train_ff_model",  # Exported
    "evaluate_ff_model",  # Exported
    # CaFo components
    "train_cafo_block_and_predictor",
    "train_cafo_model",
    "evaluate_cafo_model",
    # MF components (Corrected)
    "train_mf_hidden_layer",
    "train_mf_output_layer",
    "train_mf_model",
    "evaluate_mf_model",
    "mf_local_loss_fn",
    # Factories (Optional)
    # "get_training_function",
    # "get_evaluation_function",
]
