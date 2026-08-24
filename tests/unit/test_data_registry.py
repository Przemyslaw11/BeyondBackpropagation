from unittest.mock import patch

import pytest

from beyond_backprop.config import load_experiment_config
from beyond_backprop.data import (
    DATASET_REGISTRY,
    build_dataloaders,
    get_dataset_spec,
    get_transforms,
)


def test_dataset_registry_preserves_repository_facts() -> None:
    assert set(DATASET_REGISTRY) == {
        get_dataset_spec("MNIST").name,
        get_dataset_spec("FashionMNIST").name,
        get_dataset_spec("CIFAR10").name,
        get_dataset_spec("CIFAR100").name,
    }
    assert get_dataset_spec("MNIST").mnist_fixed_split
    assert get_dataset_spec("CIFAR100").num_classes == 100
    assert get_dataset_spec("CIFAR100").input_channels == 3


def test_canonical_preprocessing_preserves_train_only_cifar_augmentation() -> None:
    assert len(get_transforms("cifar10", train=True).transforms) == 4
    assert len(get_transforms("cifar10", train=False).transforms) == 2
    assert len(get_transforms("mnist", train=True).transforms) == 2


def test_legacy_preprocessing_import_path_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        import src.data_utils.preprocessing  # noqa: F401


def test_loader_adapter_preserves_ff_default_batch_size_and_download_policy() -> None:
    config = {
        "algorithm": {"name": "FF"},
        "general": {"backend": "local", "seed": 11},
        "data": {"name": "MNIST", "root": "./offline-data", "download": False},
    }
    with patch(
        "beyond_backprop.data.loaders.get_dataloaders", return_value=(1, None, 3)
    ) as loader:
        assert build_dataloaders(config) == (1, None, 3)

    kwargs = loader.call_args.kwargs
    assert kwargs["batch_size"] == 100
    assert kwargs["download"] is False
    assert kwargs["backend"] == "local"


def test_typed_loader_adapter_uses_resolved_experiment_values() -> None:
    config = load_experiment_config("configs/mf/mnist_mlp_2x1000.yaml")
    with patch(
        "beyond_backprop.data.loaders.get_dataloaders", return_value=(1, 2, 3)
    ) as loader:
        assert build_dataloaders(config) == (1, 2, 3)

    kwargs = loader.call_args.kwargs
    assert kwargs["dataset_name"] == "mnist"
    assert kwargs["batch_size"] == 128
    assert kwargs["download"] is True
