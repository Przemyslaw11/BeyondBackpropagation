# File: src/architectures/__init__.py
from .ff_hinton import FF_Hinton_MLP # <<< ADDED Import
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from .mf_mlp import MF_MLP

# Optional: Define a factory function or dictionary to get models by name
# from torch import nn
# from typing import Dict, Any
# def get_architecture(name: str, config: Dict[str, Any]) -> nn.Module:
#     name = name.lower()
#     model_params = config.get('model', {}).get('params', {})
#     dataset_params = config.get('data', {}) # Use 'data' section
#     # <<< Check if device is needed by constructors >>>
#     device_preference = config.get("general", {}).get("device", "auto").lower()
#     if device_preference == "cuda" and torch.cuda.is_available(): device = torch.device("cuda")
#     elif device_preference == "cpu": device = torch.device("cpu")
#     else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --- Add necessary default parameters ---
#     num_classes = dataset_params.get('num_classes', 10)
#     input_channels = dataset_params.get('input_channels', 1)
#     image_size = dataset_params.get('image_size', 28)

#     if 'num_classes' not in model_params:
#         model_params['num_classes'] = num_classes

#     if name in ['mf_mlp']: # <<< MODIFIED: Excluded ff_hinton_mlp from input_dim calc here >>>
#          if 'input_dim' not in model_params:
#              model_params['input_dim'] = input_channels * image_size * image_size
#     elif name == 'cafo_cnn':
#          if 'input_channels' not in model_params:
#              model_params['input_channels'] = input_channels
#          if 'image_size' not in model_params:
#              model_params['image_size'] = image_size
#     # FF_Hinton_MLP calculates input_dim internally based on config or uses default
#     # -----------------------------------------

#     if name == 'ff_hinton_mlp': # <<< ADDED: Handle new model >>>
#         # FF_Hinton_MLP constructor now takes full config and device
#         return FF_Hinton_MLP(config=config, device=device)
#     elif name == 'cafo_cnn':
#         return CaFo_CNN(**model_params)
#     elif name == 'mf_mlp':
#         return MF_MLP(**model_params)
#     else:
#         raise ValueError(f"Unknown architecture name: {name}")


__all__ = [
    "FF_Hinton_MLP", # <<< ADDED Export
    # "FF_Layer", # This is now internal to FF_Hinton_MLP logic conceptually
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    # "get_architecture" # Uncomment if factory is added
]