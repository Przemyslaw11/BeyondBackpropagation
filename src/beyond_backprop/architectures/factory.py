"""Canonical model factories.

Native models are still imported from the legacy implementation during the
incremental migration. Fair BP baselines are built here so algorithm-specific
heads and MF projection matrices cannot accidentally enter the comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from src.architectures.cafo_cnn import CaFo_CNN
from src.architectures.ff_mlp import FF_MLP
from src.architectures.mf_mlp import MF_MLP

from ..config.models import ArchitectureName, ExperimentConfig


def _as_mapping(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, ExperimentConfig):
        return config.to_mapping()
    return dict(config)


def _standard_mlp(
    params: Mapping[str, Any], input_dim: int, num_classes: int
) -> nn.Module:
    hidden_dims = [int(value) for value in params.get("hidden_dims", [])]
    if not hidden_dims:
        raise ValueError("BP MLP baseline requires model.params.hidden_dims")
    activation_name = str(params.get("activation", "relu")).lower()
    activation: type[nn.Module]
    if activation_name == "relu":
        activation = nn.ReLU
    elif activation_name == "tanh":
        activation = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation for BP baseline: {activation_name}")
    use_bias = bool(params.get("bias", True))
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim, bias=use_bias), activation()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
    return nn.Sequential(*layers)


def _cafo_bp_baseline(params: Mapping[str, Any], device: torch.device) -> nn.Module:
    native = CaFo_CNN(**dict(params))
    native.to(device)
    # CaFo_CNN has already computed dimensions with the corrected, state-safe
    # block probe. The probe below is kept independent and never updates BN.
    previous_mode = native.training
    native.eval()
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            int(params["input_channels"]),
            int(params["image_size"]),
            int(params["image_size"]),
            device=device,
        )
        features = native.forward_blocks_only(dummy)
        flattened_dim = int(features.flatten(1).shape[1])
    native.train(previous_mode)
    baseline = nn.Sequential(
        *native.blocks,
        nn.Flatten(),
        nn.Linear(flattened_dim, int(params["num_classes"])),
    )
    # Modules moved out of an eval-mode probing model retain that mode when
    # registered in the new Sequential. BP training must start in train mode.
    return baseline.train()


def build_fair_bp_baseline(
    config: ExperimentConfig | Mapping[str, Any],
    device: torch.device | None = None,
) -> nn.Module:
    """Build a BP model with the same native dimensions and no local-only parts."""

    mapping = _as_mapping(config)
    model_cfg = mapping.get("model", {})
    params = dict(model_cfg.get("params", {}))
    data_cfg = mapping.get("data", {})
    architecture = ArchitectureName.parse(model_cfg.get("name", "mf_mlp"))
    resolved_device = device or torch.device("cpu")
    input_dim = int(
        params.get(
            "input_dim",
            int(data_cfg.get("input_channels", 1))
            * int(data_cfg.get("image_size", 28)) ** 2,
        )
    )
    num_classes = int(data_cfg.get("num_classes", 10))
    if architecture in {ArchitectureName.FF_MLP, ArchitectureName.MF_MLP}:
        return _standard_mlp(params, input_dim, num_classes).to(resolved_device)
    if architecture is ArchitectureName.CAFO_CNN:
        params.setdefault("input_channels", int(data_cfg.get("input_channels", 1)))
        params.setdefault("image_size", int(data_cfg.get("image_size", 28)))
        params.setdefault("num_classes", num_classes)
        return _cafo_bp_baseline(params, resolved_device)
    raise ValueError(f"No BP baseline builder registered for {architecture.value}")


def build_model(
    config: ExperimentConfig | Mapping[str, Any],
    device: torch.device | None = None,
    *,
    for_bp_baseline: bool | None = None,
) -> nn.Module:
    """Build either a native algorithm model or its BP baseline."""

    mapping = _as_mapping(config)
    algorithm = str(mapping.get("algorithm", {}).get("name", "bp")).lower()
    if for_bp_baseline is None:
        for_bp_baseline = algorithm == "bp"
    if for_bp_baseline:
        return build_fair_bp_baseline(mapping, device)
    model_cfg = mapping.get("model", {})
    params = dict(model_cfg.get("params", {}))
    data_cfg = mapping.get("data", {})
    architecture = ArchitectureName.parse(model_cfg.get("name", "mf_mlp"))
    resolved_device = device or torch.device("cpu")
    params.setdefault("num_classes", int(data_cfg.get("num_classes", 10)))
    params.setdefault("input_channels", int(data_cfg.get("input_channels", 1)))
    params.setdefault("image_size", int(data_cfg.get("image_size", 28)))
    if architecture is ArchitectureName.FF_MLP:
        return FF_MLP(config=mapping, device=resolved_device, **params).to(
            resolved_device
        )
    if architecture is ArchitectureName.MF_MLP:
        params.setdefault(
            "input_dim",
            int(data_cfg.get("input_channels", 1))
            * int(data_cfg.get("image_size", 28)) ** 2,
        )
        return MF_MLP(**params).to(resolved_device)
    if architecture is ArchitectureName.CAFO_CNN:
        return CaFo_CNN(**params).to(resolved_device)
    raise ValueError(f"No native model builder registered for {architecture.value}")


class ArchitectureRegistry:
    """Small explicit registry used by callers that need custom factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str, factory: Callable[..., nn.Module]) -> None:
        key = name.strip().lower()
        if key in self._factories:
            raise ValueError(f"Architecture already registered: {name}")
        self._factories[key] = factory

    def build(self, name: str, *args: Any, **kwargs: Any) -> nn.Module:
        try:
            return self._factories[name.strip().lower()](*args, **kwargs)
        except KeyError as exc:
            raise KeyError(f"Unknown architecture: {name}") from exc
