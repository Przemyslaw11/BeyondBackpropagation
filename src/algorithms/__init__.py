# Import the main training/evaluation functions for each algorithm
from .ff import train_ff_layer, generate_positive_negative_data, ff_loss_fn # Add train_ff_model, evaluate_ff_model when implemented
from .cafo import train_cafo_block_and_predictor, train_cafo_model, evaluate_cafo_model
from .mf import train_mf_layer, train_mf_output_layer, train_mf_model, evaluate_mf_model, mf_loss_fn

# Optional: Define a factory function or dictionary to get algorithm functions by name
# def get_training_function(name: str) -> callable:
#     name = name.lower()
#     if name == 'ff':
#         # Need the main orchestrator function here
#         # return train_ff_model
#         raise NotImplementedError("train_ff_model not fully implemented yet")
#     elif name == 'cafo':
#         return train_cafo_model
#     elif name == 'mf':
#         return train_mf_model
#     else:
#         raise ValueError(f"Unknown algorithm name for training: {name}")

# def get_evaluation_function(name: str) -> callable:
#      name = name.lower()
#      if name == 'ff':
#          # return evaluate_ff_model
#          raise NotImplementedError("evaluate_ff_model not implemented yet")
#      elif name == 'cafo':
#          return evaluate_cafo_model
#      elif name == 'mf':
#          return evaluate_mf_model
#      else:
#          raise ValueError(f"Unknown algorithm name for evaluation: {name}")


__all__ = [
    # FF components
    "train_ff_layer",
    "generate_positive_negative_data",
    "ff_loss_fn",
    # "train_ff_model", # Add when implemented
    # "evaluate_ff_model", # Add when implemented

    # CaFo components
    "train_cafo_block_and_predictor",
    "train_cafo_model",
    "evaluate_cafo_model",

    # MF components
    "train_mf_layer",
    "train_mf_output_layer",
    "train_mf_model",
    "evaluate_mf_model",
    "mf_loss_fn",

    # Factories (Optional)
    # "get_training_function",
    # "get_evaluation_function",
]
