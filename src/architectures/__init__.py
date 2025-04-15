# File: ./src/architectures/__init__.py # <<< MODIFIED >>>
# --------------------------------------------------------------------------------
# File: ./src/architectures/__init__.py (FINAL CORRECTED)
# Import the main training/evaluation functions for each algorithm

# --- FF ---
from .ff_mlp import FF_MLP # <<< NOTE: This now refers to the corrected class >>>
# Removed ff_hinton.py as its logic is merged into ff_mlp.py

# --- CaFo ---
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor

# --- MF ---
from .mf_mlp import MF_MLP

# --- Factory Functions ---
from typing import Callable, Dict, Any
import torch.nn as nn
import torch # <<< Added import
import logging # <<< Added import

logger = logging.getLogger(__name__) # <<< Added logger

def get_architecture(name: str, config: Dict[str, Any], device: torch.device) -> nn.Module: # Added device
    name = name.lower()
    model_params = config.get('model', {}).get('params', {})
    dataset_params = config.get('data', {})
    num_classes = dataset_params.get('num_classes', 10)
    input_channels = dataset_params.get('input_channels', 1)
    image_size = dataset_params.get('image_size', 28)
    # <<< Add algorithm name to determine BP baseline adaptation >>>
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    # <<< CORRECTION: Ensure num_classes is passed for all models if not in params >>>
    if 'num_classes' not in model_params: model_params['num_classes'] = num_classes

    # Calculate input_dim for MLPs if needed
    if name in ['ff_mlp', 'mf_mlp']:
         if 'input_dim' not in model_params:
             model_params['input_dim'] = input_channels * image_size * image_size
             logger.debug(f"Calculated input_dim={model_params['input_dim']} for {name}")

    # Add necessary params for CNN if needed
    elif name == 'cafo_cnn':
         if 'input_channels' not in model_params: model_params['input_channels'] = input_channels
         if 'image_size' not in model_params: model_params['image_size'] = image_size

    # Instantiate
    if name == 'ff_mlp':
        # <<< CORRECTION: Adapt FF_MLP for BP baseline explicitly here >>>
        if is_bp_baseline:
            logger.info("Adapting modified FF_MLP structure for BP baseline -> Standard nn.Sequential MLP.")
            bp_input_dim = model_params["input_dim"]
            hidden_dims = model_params.get("hidden_dims", [])
            activation_name = model_params.get("activation", "ReLU").lower()
            use_bias = model_params.get("bias", True)
            # <<< CORRECTION: Use bias_init from params, but BP doesn't need special handling >>>
            bias_init = model_params.get("bias_init", 0.0) # Read config default
            if not hidden_dims: raise ValueError("BP baseline creation failed: hidden_dims missing for FF_MLP.")
            layers = []
            current_dim = bp_input_dim
            act_cls = nn.ReLU if activation_name == 'relu' else nn.Tanh # Use standard activation
            for h_dim in hidden_dims:
                linear_layer = nn.Linear(current_dim, h_dim, bias=use_bias)
                # Apply standard Kaiming/Xavier init for BP layers
                if activation_name == 'relu': nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
                else: nn.init.xavier_normal_(linear_layer.weight)
                if use_bias: nn.init.constant_(linear_layer.bias, bias_init) # Use config bias_init
                layers.append(linear_layer)
                layers.append(act_cls())
                current_dim = h_dim
            # Add final layer for classification
            final_layer = nn.Linear(current_dim, num_classes, bias=use_bias)
            # Standard init for final BP layer
            nn.init.normal_(final_layer.weight, std=0.01)
            if use_bias: nn.init.constant_(final_layer.bias, bias_init)
            layers.append(final_layer)
            model = nn.Sequential(*layers)
            logger.debug("Created BP baseline Sequential model from modified FF_MLP spec.")
            return model # Return the Sequential model for BP
        else:
            # Instantiate the modified FF_MLP for native FF training
            # <<< CORRECTION: Pass config and device >>>
            return FF_MLP(config=config, device=device, **model_params)

    elif name == 'cafo_cnn':
        # <<< CORRECTION: Adapt CaFo_CNN for BP baseline explicitly here >>>
        cafo_base = CaFo_CNN(**model_params)
        if is_bp_baseline:
            logger.info("Creating BP baseline model from CaFo_CNN blocks + final Linear layer.")
            cafo_base.to(device)
            with torch.no_grad():
                # Use correct image size from params
                img_sz_cnn = model_params.get('image_size', 32)
                in_ch_cnn = model_params.get('input_channels', 3)
                dummy_input = torch.randn(1, in_ch_cnn, img_sz_cnn, img_sz_cnn).to(device)
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel()
            cafo_base.cpu()
            logger.debug(f"Flattened output dimension from CaFo blocks: {num_output_features}")
            # Standard init for final BP layer
            final_linear = nn.Linear(num_output_features, num_classes)
            nn.init.normal_(final_linear.weight, std=0.01)
            if final_linear.bias is not None: nn.init.zeros_(final_linear.bias)
            model = nn.Sequential(cafo_base.blocks, nn.Flatten(), final_linear)
            logger.debug("Created BP baseline Sequential model from CaFo_CNN spec.")
            return model # Return the Sequential model for BP
        else:
            # Return the native CaFo base for CaFo training
            return cafo_base

    elif name == 'mf_mlp':
         # MF_MLP can be used directly by BP baseline by calling its standard forward
        model = MF_MLP(**model_params)
        if is_bp_baseline: logger.info("Using standard forward pass of MF_MLP for BP baseline.")
        return model
    else:
        raise ValueError(f"Unknown architecture name: {name}")


__all__ = [
    "FF_MLP", # <<< CORRECTION: Keep exporting FF_MLP (the modified one) >>>
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    "get_architecture" # Export factory if used
]
# --------------------------------------------------------------------------------