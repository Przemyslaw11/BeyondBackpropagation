# File: ./src/architectures/__init__.py (Verify content - Ensure CaFo BP fix applied here too if used)
# Import the architecture classes
from .ff_mlp import FF_MLP
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from .mf_mlp import MF_MLP

# --- Factory Function (Ensure BP baseline creation is correct) ---
from typing import Dict, Any, Optional, Callable # Added Optional, Callable
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def get_architecture(
    name: str,
    config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    Factory function to instantiate the correct model architecture.
    Handles creation of BP baseline models from alternative algorithm structures.
    """
    name = name.lower()
    model_params = config.get('model', {}).get('params', {})
    dataset_params = config.get('data', {})
    num_classes = dataset_params.get('num_classes', 10)
    input_channels = dataset_params.get('input_channels', 1)
    image_size = dataset_params.get('image_size', 28)
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    logger.info(f"Getting architecture: {name} (Algo: {algorithm_name.upper()})")

    # --- Add default/calculated parameters if missing ---
    if 'num_classes' not in model_params: model_params['num_classes'] = num_classes
    if name in ['ff_mlp', 'mf_mlp']:
         if 'input_dim' not in model_params:
             model_params['input_dim'] = input_channels * image_size * image_size
             logger.debug(f"Calculated input_dim={model_params['input_dim']} for {name}")
    elif name == 'cafo_cnn':
         if 'input_channels' not in model_params: model_params['input_channels'] = input_channels
         if 'image_size' not in model_params: model_params['image_size'] = image_size

    # --- Instantiate Architecture ---
    model: Optional[nn.Module] = None
    if name == 'ff_mlp':
        if is_bp_baseline:
            logger.info("Adapting modified FF_MLP structure for BP baseline -> Standard nn.Sequential MLP.")
            bp_input_dim = model_params["input_dim"]
            hidden_dims = model_params.get("hidden_dims", [])
            activation_name = model_params.get("activation", "ReLU").lower()
            use_bias = model_params.get("bias", True)
            if not hidden_dims: raise ValueError("BP baseline creation failed: hidden_dims missing for FF_MLP.")
            layers = []
            current_dim = bp_input_dim
            act_cls = nn.ReLU if activation_name == 'relu' else nn.Tanh
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
                layers.append(act_cls())
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
            model = nn.Sequential(*layers)
            logger.debug("Created BP baseline Sequential model from modified FF_MLP spec.")
        else:
            # Pass full config and device to the constructor
            model = FF_MLP(config=config, device=device, **model_params)
            logger.debug("Using native modified FF_MLP structure.")

    elif name == 'cafo_cnn':
        cafo_base = CaFo_CNN(**model_params)
        if is_bp_baseline:
            logger.info("Creating BP baseline model from CaFo_CNN blocks + final Linear layer.")
            cafo_base.to(device)
            with torch.no_grad():
                dummy_input = torch.randn(1, model_params['input_channels'], model_params['image_size'], model_params['image_size']).to(device)
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel()
            cafo_base.cpu()
            logger.debug(f"Flattened output dimension from CaFo blocks: {num_output_features}")

            # <<< --- CORRECTED MODEL CREATION --- >>>
            model = nn.Sequential(
                *cafo_base.blocks, # Unpack the list here
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes)
            )
            # <<< --- END CORRECTION --- >>>

            logger.debug("Created BP baseline Sequential model from CaFo_CNN spec.")
        else:
            model = cafo_base
            logger.debug("Using native CaFo_CNN structure.")

    elif name == 'mf_mlp':
         model = MF_MLP(**model_params)
         if is_bp_baseline: logger.info("Using standard forward pass of MF_MLP for BP baseline.")
         else: logger.debug("Using native MF_MLP structure.")

    else:
        raise ValueError(f"Unknown architecture name: {name}")

    if model is None: raise RuntimeError("Model instantiation failed.")
    return model


__all__ = [
    "FF_MLP",
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    "get_architecture"
]