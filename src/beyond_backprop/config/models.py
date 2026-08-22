"""Frozen configuration objects and canonical enum values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar

_EnumType = TypeVar("_EnumType", bound="_ValueEnum")


class _ValueEnum(str, Enum):
    @classmethod
    def parse(cls: type[_EnumType], value: str | Enum) -> _EnumType:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        for member in cls:
            if member.value == normalized:
                return member
        values = ", ".join(member.value for member in cls)
        raise ValueError(
            f"Unsupported {cls.__name__} value {value!r}; expected one of {values}"
        )


class AlgorithmName(_ValueEnum):
    BP = "bp"
    FF = "ff"
    CAFO = "cafo"
    MF = "mf"


class ArchitectureName(_ValueEnum):
    FF_MLP = "ff_mlp"
    MF_MLP = "mf_mlp"
    CAFO_CNN = "cafo_cnn"


class DatasetName(_ValueEnum):
    MNIST = "mnist"
    FASHION_MNIST = "fashionmnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

    @classmethod
    def parse(cls, value: str | Enum) -> DatasetName:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("_", "")
        return cls(normalized)


class BackendName(_ValueEnum):
    LOCAL = "local"
    SLURM = "slurm"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return ordinary mutable containers for compatibility APIs and YAML."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated, immutable view of the merged experiment configuration."""

    experiment_name: str
    algorithm: AlgorithmName
    architecture: ArchitectureName
    dataset: DatasetName
    backend: BackendName
    device: str
    seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    data_root: str
    download: bool
    val_split: float
    num_classes: int
    input_channels: int
    image_size: int
    model_params: Mapping[str, Any] = field(default_factory=dict)
    algorithm_params: Mapping[str, Any] = field(default_factory=dict)
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    monitoring: Mapping[str, Any] = field(default_factory=dict)
    tracking: Mapping[str, Any] = field(default_factory=dict)
    resolved: Mapping[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "model_params",
            "algorithm_params",
            "optimizer",
            "training",
            "monitoring",
            "tracking",
            "resolved",
        ):
            object.__setattr__(
                self, field_name, _freeze(dict(getattr(self, field_name)))
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return a mutable copy suitable for legacy compatibility APIs."""

        return _thaw(self.resolved)
