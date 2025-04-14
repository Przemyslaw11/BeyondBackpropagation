# File: src/architectures/__init__.py
# Remove FF_Layer export if it exists, keep FF_MLP
from .ff_mlp import FF_MLP # This now refers to the modified class
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from .mf_mlp import MF_MLP

# Optional: Factory function (Needs update if used)
from torch import nn
import torch # <<< Added import
from typing import Dict, Any
def get_architecture(name: str, config: Dict[str, Any], device: torch.device) -> nn.Module: # Added device
    name = name.lower()
    model_params = config.get('model', {}).get('params', {})
    dataset_params = config.get('data', {})
    num_classes = dataset_params.get('num_classes', 10)
    input_channels = dataset_params.get('input_channels', 1)
    image_size = dataset_params.get('image_size', 28)

    if 'num_classes' not in model_params: model_params['num_classes'] = num_classes

    # Calculate input_dim for MLPs if needed
    if name in ['ff_mlp', 'mf_mlp']:
         if 'input_dim' not in model_params: model_params['input_dim'] = input_channels * image_size * image_size

    # Add necessary params for CNN if needed
    elif name == 'cafo_cnn':
         if 'input_channels' not in model_params: model_params['input_channels'] = input_channels
         if 'image_size' not in model_params: model_params['image_size'] = image_size

    # Instantiate
    if name == 'ff_mlp':
        # Modified FF_MLP now takes full config and device
        return FF_MLP(config=config, device=device, **model_params) # Pass params too
    elif name == 'cafo_cnn':
        return CaFo_CNN(**model_params)
    elif name == 'mf_mlp':
        return MF_MLP(**model_params)
    else:
        raise ValueError(f"Unknown architecture name: {name}")


__all__ = [
    "FF_MLP", # Keep exporting FF_MLP (the modified one)
    # "FF_Layer", # Remove if it was exported
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    "get_architecture" # Export factory if used
]