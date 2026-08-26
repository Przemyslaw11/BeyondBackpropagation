"""Dependency-free model FLOP estimates with explicit provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _input_tensor(config: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    data = config.get("data", {})
    channels = int(data.get("input_channels", 1))
    size = int(data.get("image_size", 28))
    return torch.zeros(1, channels, size, size, device=device)


def profile_model(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Estimate forward and BP-update GFLOPs without changing model state."""

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    forward_flops = 0
    hooks = []

    def linear_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        if inputs and isinstance(inputs[0], torch.Tensor):
            forward_flops_local = (
                2 * inputs[0].shape[0] * module.in_features * module.out_features
            )
            nonlocal forward_flops
            forward_flops += int(forward_flops_local)

    def conv_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        if inputs and isinstance(inputs[0], torch.Tensor):
            batch = inputs[0].shape[0]
            # The output is captured through the closure below; this hook is
            # installed with ``with_kwargs=False`` for all supported PyTorch versions.
            nonlocal forward_flops
            if isinstance(module, nn.Conv2d):
                output_elements = batch * module.out_channels
                # Spatial dimensions are reconstructed from the input and conv parameters.
                height = (
                    inputs[0].shape[2]
                    + 2 * module.padding[0]
                    - module.dilation[0] * (module.kernel_size[0] - 1)
                    - 1
                ) // module.stride[0] + 1
                width = (
                    inputs[0].shape[3]
                    + 2 * module.padding[1]
                    - module.dilation[1] * (module.kernel_size[1] - 1)
                    - 1
                ) // module.stride[1] + 1
                forward_flops += int(
                    2
                    * output_elements
                    * height
                    * width
                    * module.in_channels
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))

    was_training = model.training
    model.eval()
    sample = _input_tensor(config, device)
    try:
        with torch.no_grad():
            try:
                model(sample)
            except (RuntimeError, TypeError):
                model(sample.flatten(1))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    forward_gflops = forward_flops / 1e9
    return {
        "model_parameters_trainable": {
            "value": trainable,
            "unit": "parameters",
            "source": "model-introspection",
            "measured": True,
        },
        "model_parameters_total": {
            "value": total,
            "unit": "parameters",
            "source": "model-introspection",
            "measured": True,
        },
        "forward_gflops": {
            "value": forward_gflops,
            "unit": "GFLOPs",
            "source": "estimated-forward-hooks",
            "measured": False,
        },
        "estimated_fwd_gflops": {
            "value": forward_gflops,
            "unit": "GFLOPs",
            "source": "estimated-forward-hooks",
            "measured": False,
        },
        "estimated_bp_update_gflops": {
            "value": forward_gflops * 2.0,
            "unit": "GFLOPs",
            "source": "estimated-forward-hooks-x2",
            "measured": False,
        },
    }


__all__ = ["profile_model"]
