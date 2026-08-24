"""Explicit dataset registry without importing torchvision at package import time."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import DatasetName


@dataclass(frozen=True)
class DatasetSpec:
    """Scientific dataset facts used by validation and model construction."""

    name: DatasetName
    torchvision_name: str
    num_classes: int
    input_channels: int
    image_size: int
    mnist_fixed_split: bool = False


DATASET_REGISTRY: dict[DatasetName, DatasetSpec] = {
    DatasetName.MNIST: DatasetSpec(
        DatasetName.MNIST, "MNIST", 10, 1, 28, mnist_fixed_split=True
    ),
    DatasetName.FASHION_MNIST: DatasetSpec(
        DatasetName.FASHION_MNIST, "FashionMNIST", 10, 1, 28
    ),
    DatasetName.CIFAR10: DatasetSpec(DatasetName.CIFAR10, "CIFAR10", 10, 3, 32),
    DatasetName.CIFAR100: DatasetSpec(DatasetName.CIFAR100, "CIFAR100", 100, 3, 32),
}


def get_dataset_spec(name: DatasetName | str) -> DatasetSpec:
    """Resolve a dataset name and fail with an actionable registry error."""

    dataset_name = DatasetName.parse(name)
    try:
        return DATASET_REGISTRY[dataset_name]
    except KeyError as exc:  # pragma: no cover - registry and enum are kept together
        raise ValueError(f"Dataset is not registered: {dataset_name.value}") from exc
