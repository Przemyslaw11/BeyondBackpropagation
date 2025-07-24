"""Architecture-related modules, including model definitions and a factory."""

import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from .ff_mlp import FF_MLP
from .mf_mlp import MF_MLP

logger = logging.getLogger(__name__)


def get_architecture(
    name: str,
    config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Factory function to instantiate the correct model architecture.

    Handles the creation of BP baseline models from the structures of
    alternative algorithms (FF, MF, CaFo).
    """
    name = name.lower()
    model_config = config.get("model", {})
    arch_params = model_config.get("params", {})
    dataset_config = config.get("data", {})
    num_classes = dataset_config.get("num_classes", 10)
    input_channels = dataset_config.get("input_channels", 1)
    image_size = dataset_config.get("image_size", 28)
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    logger.info("Getting architecture: %s (Algo: %s)", name, algorithm_name.upper())

    if "num_classes" not in arch_params:
        arch_params["num_classes"] = num_classes
    if name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
            logger.debug(
                "Calculated input_dim=%d for %s", arch_params["input_dim"], name
            )
    elif name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
    else:
        if name not in ["ff_mlp", "mf_mlp", "cafo_cnn"]:
            raise ValueError(f"Unknown architecture base name: {name}")

    model: Optional[nn.Module] = None
    input_adapter: Optional[Callable] = None

    if name in ["ff_mlp", "mf_mlp"]:

        def flatten_input(x: torch.Tensor) -> torch.Tensor:
            return x.view(x.shape[0], -1)

        input_adapter = flatten_input
        logger.debug("Arch %s requires input flattening adapter.", name)
        if name == "ff_mlp":
            if is_bp_baseline:
                logger.info(
                    "Adapting FF_MLP structure for BP baseline -> nn.Sequential MLP."
                )
                bp_input_dim = arch_params["input_dim"]
                hidden_dims = arch_params.get("hidden_dims", [])
                activation_name = arch_params.get("activation", "ReLU").lower()
                use_bias = arch_params.get("bias", True)
                if not hidden_dims:
                    raise ValueError(
                        "BP baseline failed: hidden_dims missing for FF_MLP."
                    )
                layers = []
                current_dim = bp_input_dim
                act_cls = nn.ReLU if activation_name == "relu" else nn.Tanh
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
                    layers.append(act_cls())
                    current_dim = h_dim
                layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
                model = nn.Sequential(*layers)
                logger.debug("Created BP baseline from modified FF_MLP spec.")
            else:
                model = FF_MLP(config=config, device=device, **arch_params)
                logger.debug("Using native modified FF_MLP structure.")
        elif name == "mf_mlp":
            model = MF_MLP(**arch_params)
            if is_bp_baseline:
                logger.info("Using standard forward pass of MF_MLP for BP baseline.")
            else:
                logger.debug("Using native MF_MLP structure.")

    elif name == "cafo_cnn":
        input_adapter = None
        logger.debug("Arch %s does not require standard input adapter.", name)

        cnn_base = CaFo_CNN(**arch_params)

        if is_bp_baseline:
            logger.info(
                "Creating BP baseline from %s blocks + Linear layer.", name.upper()
            )
            cnn_base.to(device)
            with torch.no_grad():
                dummy_input_shape = (
                    1,
                    arch_params["input_channels"],
                    arch_params["image_size"],
                    arch_params["image_size"],
                )
                dummy_input = torch.randn(dummy_input_shape).to(device)

                model_blocks = cnn_base.blocks
                dummy_features = dummy_input
                for block in model_blocks:
                    dummy_features = block(dummy_features)
                num_output_features = dummy_features.numel()
            cnn_base.cpu()
            logger.debug(
                "Flattened output dimension from %s blocks: %d",
                name.upper(),
                num_output_features,
            )

            model = nn.Sequential(
                *cnn_base.blocks,
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes),
            )
            logger.debug("Created BP baseline from %s spec.", name.upper())
        else:
            model = cnn_base
            logger.debug("Using native %s structure.", name.upper())

    else:
        if name not in ["ff_mlp", "mf_mlp", "cafo_cnn"]:
            raise ValueError(f"Unknown or unhandled architecture name: {name}")

    if model is None:
        raise RuntimeError(f"Model instantiation failed for architecture: {name}")

    logger.info("Model '%s' (Algo: %s) created.", name.upper(), algorithm_name.upper())
    return model, input_adapter


__all__ = [
    "FF_MLP",
    "MF_MLP",
    "CaFoBlock",
    "CaFoPredictor",
    "CaFo_CNN",
    "get_architecture",
]
