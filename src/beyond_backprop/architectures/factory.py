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

from ..config.models import ArchitectureName, ExperimentConfig
from .cafo_cnn import CaFo_CNN
from .ff_mlp import FF_MLP
from .mf_mlp import MF_MLP


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


ArchitectureFactory = Callable[[Mapping[str, Any], torch.device], nn.Module]


class ArchitectureRegistry:
    """Registry for native models and their fair BP counterparts."""

    def __init__(self) -> None:
        self._native: dict[str, ArchitectureFactory] = {}
        self._baselines: dict[str, ArchitectureFactory] = {}

    def register(
        self,
        name: str,
        native_factory: ArchitectureFactory,
        baseline_factory: ArchitectureFactory | None = None,
    ) -> None:
        key = name.strip().lower().replace("-", "_")
        if key in self._native:
            raise ValueError(f"Architecture already registered: {name}")
        self._native[key] = native_factory
        if baseline_factory is not None:
            self._baselines[key] = baseline_factory

    def build(self, name: str, *args: Any, **kwargs: Any) -> nn.Module:
        """Build a native model for compatibility with the initial registry API."""

        return self.build_native(name, *args, **kwargs)

    def build_native(
        self, name: str, config: Mapping[str, Any], device: torch.device
    ) -> nn.Module:
        return self._lookup(self._native, name)(config, device)

    def build_baseline(
        self, name: str, config: Mapping[str, Any], device: torch.device
    ) -> nn.Module:
        return self._lookup(self._baselines, name)(config, device)

    @staticmethod
    def _lookup(
        factories: Mapping[str, ArchitectureFactory], name: str
    ) -> ArchitectureFactory:
        key = name.strip().lower().replace("-", "_")
        try:
            return factories[key]
        except KeyError as exc:
            raise KeyError(f"Unknown architecture: {name}") from exc


def _model_params(mapping: Mapping[str, Any]) -> dict[str, Any]:
    model_cfg = mapping.get("model", {})
    return dict(model_cfg.get("params", {}))


def _build_ff_native(mapping: Mapping[str, Any], device: torch.device) -> nn.Module:
    params = _model_params(mapping)
    params["num_classes"] = int(mapping.get("data", {}).get("num_classes", 10))
    return FF_MLP(config=dict(mapping), device=device, **params).to(device)


def _build_mf_native(mapping: Mapping[str, Any], device: torch.device) -> nn.Module:
    params = _model_params(mapping)
    data_cfg = mapping.get("data", {})
    params["num_classes"] = int(data_cfg.get("num_classes", 10))
    params.setdefault(
        "input_dim",
        int(data_cfg.get("input_channels", 1))
        * int(data_cfg.get("image_size", 28)) ** 2,
    )
    return MF_MLP(**params).to(device)


def _build_cafo_native(mapping: Mapping[str, Any], device: torch.device) -> nn.Module:
    params = _model_params(mapping)
    data_cfg = mapping.get("data", {})
    params.update(
        {
            "input_channels": int(data_cfg.get("input_channels", 1)),
            "image_size": int(data_cfg.get("image_size", 28)),
            "num_classes": int(data_cfg.get("num_classes", 10)),
        }
    )
    return CaFo_CNN(**params).to(device)


def _build_mlp_baseline(mapping: Mapping[str, Any], device: torch.device) -> nn.Module:
    params = _model_params(mapping)
    data_cfg = mapping.get("data", {})
    input_dim = int(
        params.get(
            "input_dim",
            int(data_cfg.get("input_channels", 1))
            * int(data_cfg.get("image_size", 28)) ** 2,
        )
    )
    return _standard_mlp(params, input_dim, int(data_cfg.get("num_classes", 10))).to(
        device
    )


def _build_cafo_baseline(mapping: Mapping[str, Any], device: torch.device) -> nn.Module:
    params = _model_params(mapping)
    data_cfg = mapping.get("data", {})
    params.update(
        {
            "input_channels": int(data_cfg.get("input_channels", 1)),
            "image_size": int(data_cfg.get("image_size", 28)),
            "num_classes": int(data_cfg.get("num_classes", 10)),
        }
    )
    return _cafo_bp_baseline(params, device)


ARCHITECTURE_REGISTRY = ArchitectureRegistry()
ARCHITECTURE_REGISTRY.register(
    ArchitectureName.FF_MLP.value, _build_ff_native, _build_mlp_baseline
)
ARCHITECTURE_REGISTRY.register(
    ArchitectureName.MF_MLP.value, _build_mf_native, _build_mlp_baseline
)
ARCHITECTURE_REGISTRY.register(
    ArchitectureName.CAFO_CNN.value, _build_cafo_native, _build_cafo_baseline
)


def build_fair_bp_baseline(
    config: ExperimentConfig | Mapping[str, Any],
    device: torch.device | None = None,
) -> nn.Module:
    """Build a BP model with the same native dimensions and no local-only parts."""

    mapping = _as_mapping(config)
    model_cfg = mapping.get("model", {})
    architecture = ArchitectureName.parse(model_cfg.get("name", "mf_mlp"))
    resolved_device = device or torch.device("cpu")
    return ARCHITECTURE_REGISTRY.build_baseline(
        architecture.value, mapping, resolved_device
    )


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
    architecture = ArchitectureName.parse(model_cfg.get("name", "mf_mlp"))
    resolved_device = device or torch.device("cpu")
    return ARCHITECTURE_REGISTRY.build_native(
        architecture.value, mapping, resolved_device
    )
