# File: src/architectures/__init__.py
from .ff_mlp import FF_MLP, FF_Layer
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from .mf_mlp import MF_MLP

# Optional: Define a factory function or dictionary to get models by name
# from torch import nn
# from typing import Dict, Any
# def get_architecture(name: str, config: Dict[str, Any]) -> nn.Module:
#     name = name.lower()
#     model_params = config.get('model', {}).get('params', {})
#     dataset_params = config.get('data', {}) # Use 'data' section

#     # --- Add necessary default parameters ---
#     num_classes = dataset_params.get('num_classes', 10)
#     input_channels = dataset_params.get('input_channels', 1)
#     image_size = dataset_params.get('image_size', 28)

#     if 'num_classes' not in model_params:
#         model_params['num_classes'] = num_classes

#     if name in ['ff_mlp', 'mf_mlp']:
#          if 'input_dim' not in model_params:
#              model_params['input_dim'] = input_channels * image_size * image_size
#     elif name == 'cafo_cnn':
#          if 'input_channels' not in model_params:
#              model_params['input_channels'] = input_channels
#          if 'image_size' not in model_params:
#              model_params['image_size'] = image_size
#     # -----------------------------------------

#     if name == 'ff_mlp':
#         return FF_MLP(**model_params)
#     elif name == 'cafo_cnn':
#         return CaFo_CNN(**model_params)
#     elif name == 'mf_mlp':
#         return MF_MLP(**model_params)
#     else:
#         raise ValueError(f"Unknown architecture name: {name}")


__all__ = [
    "FF_MLP",
    "FF_Layer",
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    # "get_architecture" # Uncomment if factory is added
]
