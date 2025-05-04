# File: ./src/architectures/__init__.py (MODIFIED)
# Import the architecture classes
from .ff_mlp import FF_MLP
from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from .mf_mlp import MF_MLP
from .mf_cnn import MF_CNN # <<< NEW >>>

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
    model_config = config.get('model', {})
    arch_params = model_config.get('params', {})
    dataset_config = config.get('data', {})
    num_classes = dataset_config.get('num_classes', 10)
    input_channels = dataset_config.get('input_channels', 1)
    image_size = dataset_config.get('image_size', 28)
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    logger.info(f"Getting architecture: {name} (Algo: {algorithm_name.upper()})")

    # --- Add default/calculated parameters if missing ---
    if 'num_classes' not in arch_params: arch_params['num_classes'] = num_classes
    if name in ['ff_mlp', 'mf_mlp']:
         if 'input_dim' not in arch_params:
             arch_params['input_dim'] = input_channels * image_size * image_size
             logger.debug(f"Calculated input_dim={arch_params['input_dim']} for {name}")
    # <<< MODIFIED: Added 'mf_cnn' >>>
    elif name in ['cafo_cnn', 'mf_cnn']:
         if 'input_channels' not in arch_params: arch_params['input_channels'] = input_channels
         if 'image_size' not in arch_params: arch_params['image_size'] = image_size
    else:
        raise ValueError(f"Unknown architecture base name: {name}")

    # --- Instantiate Architecture ---
    model: Optional[nn.Module] = None
    # <<< MODIFIED: Consolidate MLP logic >>>
    if name in ['ff_mlp', 'mf_mlp']:
        input_adapter = lambda x: x.view(x.shape[0], -1) # Flatten input
        logger.debug(f"Arch {name} requires input flattening adapter.")
        if name == 'ff_mlp':
            if is_bp_baseline:
                logger.info("Adapting modified FF_MLP structure for BP baseline -> Standard nn.Sequential MLP.")
                bp_input_dim = arch_params["input_dim"]
                hidden_dims = arch_params.get("hidden_dims", [])
                activation_name = arch_params.get("activation", "ReLU").lower()
                use_bias = arch_params.get("bias", True)
                if not hidden_dims: raise ValueError("BP baseline creation failed: hidden_dims missing for FF_MLP.")
                layers = []
                current_dim = bp_input_dim
                act_cls = nn.ReLU if activation_name == 'relu' else nn.Tanh # Use standard activation
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
                    layers.append(act_cls())
                    current_dim = h_dim
                layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
                model = nn.Sequential(*layers)
                logger.debug("Created BP baseline Sequential model from modified FF_MLP spec.")
            else:
                model = FF_MLP(config=config, device=device, **arch_params)
                logger.debug("Using native modified FF_MLP structure.")
        elif name == 'mf_mlp':
            model = MF_MLP(**arch_params)
            if is_bp_baseline: logger.info("Using standard forward pass of MF_MLP for BP baseline.")
            else: logger.debug("Using native MF_MLP structure.")

    # <<< MODIFIED: Consolidate CNN logic >>>
    elif name in ['cafo_cnn', 'mf_cnn']:
        input_adapter = None # CNNs don't need flattening adapter
        logger.debug(f"Arch {name} does not require standard input adapter.")
        if name == 'cafo_cnn':
            cnn_base = CaFo_CNN(**arch_params) # Instantiate CaFo base
        elif name == 'mf_cnn':
            cnn_base = MF_CNN(**arch_params) # Instantiate MF base
        else: # Should not happen due to initial check
             raise ValueError(f"Internal error: Unexpected CNN arch name {name}")

        if is_bp_baseline:
            logger.info(f"Creating BP baseline model from {name.upper()} blocks + final Linear layer.")
            # Determine final flattened dimension dynamically
            cnn_base.to(device)
            with torch.no_grad():
                 # Use a shape that matches the config
                 dummy_input_shape = (1, arch_params['input_channels'], arch_params['image_size'], arch_params['image_size'])
                 dummy_input = torch.randn(dummy_input_shape).to(device)
                 # Use the BP-specific forward pass to get final features before classifier
                 features = cnn_base(dummy_input) # Assumes forward() returns logits
                 # Need to get features *before* the final linear layer
                 # Let's redefine the BP baseline structure more robustly
                 model_blocks = cnn_base.blocks
                 dummy_features = dummy_input
                 for block in model_blocks:
                     dummy_features = block(dummy_features)
                 num_output_features = dummy_features.numel() # Flattened size
            cnn_base.cpu() # Move base back if needed
            logger.debug(f"Flattened output dimension from {name.upper()} blocks: {num_output_features}")

            # Create the BP baseline as Sequential
            model = nn.Sequential(
                *cnn_base.blocks, # Unpack the ModuleList of blocks
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes) # Add the final classifier
            )
            logger.debug(f"Created BP baseline Sequential model from {name.upper()} spec.")
        else:
            model = cnn_base # Use the native MF_CNN or CaFo_CNN structure
            logger.debug(f"Using native {name.upper()} structure.")

    else: # Should not be reachable if initial checks are correct
        raise ValueError(f"Unknown or unhandled architecture name: {name}")

    if model is None: raise RuntimeError("Model instantiation failed.")
    logger.info(f"Model '{name.upper()}' (Algo: {algorithm_name.upper()}) created.")
    # <<< RETURN MODIFIED >>> Return adapter as well
    return model, input_adapter


# <<< MODIFIED >>> Update __all__
__all__ = [
    "FF_MLP",
    "CaFo_CNN",
    "CaFoBlock",
    "CaFoPredictor",
    "MF_MLP",
    "MF_CNN", # Add new class
    "get_architecture" # Keep factory
]