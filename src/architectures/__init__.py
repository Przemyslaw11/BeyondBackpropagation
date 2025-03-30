# File: src/architectures/__init__.py
from .ff_mlp import FF_MLP, FF_Layer  # Expose FF_Layer if needed
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor  # Expose block/predictor
from .mf_mlp import MF_MLP

# Optional: Define a factory function or dictionary to get models by name
# from torch import nn
# from typing import Dict, Any
# def get_architecture(name: str, config: Dict[str, Any]) -> nn.Module:
#     name = name.lower()
#     model_params = config.get('model', {}).get('params', {})
#     dataset_params = config.get('dataset', {})

#     # Add necessary default parameters from dataset_params if needed
#     if name in ['ff_mlp', 'mf_mlp']:
#          if 'input_dim' not in model_params:
#              model_params['input_dim'] = dataset_params.get('input_channels', 1) * dataset_params.get('image_size', 28)**2
#          if 'num_classes' not in model_params:
#              model_params['num_classes'] = dataset_params.get('num_classes', 10)
#     elif name == 'cafo_cnn':
#          if 'input_channels' not in model_params:
#              model_params['input_channels'] = dataset_params.get('input_channels', 1)
#          if 'image_size' not in model_params:
#              model_params['image_size'] = dataset_params.get('image_size', 28)
#          if 'num_classes' not in model_params:
#              model_params['num_classes'] = dataset_params.get('num_classes', 10)

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
