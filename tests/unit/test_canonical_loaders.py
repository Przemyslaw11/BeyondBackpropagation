from __future__ import annotations

from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import beyond_backprop.runtime.backend_policy as backend_policy
from beyond_backprop.data import build_dataloaders, get_transforms
from beyond_backprop.data import loaders as canonical_loaders


class FakeVisionDataset(Dataset):
    calls: list[dict[str, object]] = []
    lengths = {True: 60000, False: 100}

    def __init__(self, *, root, train, download, transform=None):
        self.calls.append(
            {
                "root": root,
                "train": train,
                "download": download,
                "transform": transform,
            }
        )
        self.train = train
        self.transform = transform

    def __len__(self):
        return self.lengths[self.train]

    def __getitem__(self, index):
        image = torch.zeros(1, 28, 28)
        if self.transform is not None:
            image = self.transform(image)
        return image, index % 10


def _patch_dataset(name: str):
    return patch.object(canonical_loaders.torchvision.datasets, name, FakeVisionDataset)


def test_mnist_fixed_split_and_evaluation_transform_isolation():
    FakeVisionDataset.calls = []
    with _patch_dataset("MNIST"):
        train, validation, test = canonical_loaders.get_dataloaders(
            "MNIST",
            batch_size=100,
            val_split=0.2,
            seed=7,
            backend="local",
            num_workers=0,
            pin_memory=False,
            download=False,
        )

    assert validation is not None
    assert len(train.dataset) == 50000
    assert len(validation.dataset) == 10000
    assert train.dataset.subset.indices[:2] == [0, 1]
    assert validation.dataset.subset.indices[:2] == [50000, 50001]
    assert train.dataset.transform is not validation.dataset.transform
    assert FakeVisionDataset.calls[0]["download"] is False
    assert FakeVisionDataset.calls[1]["download"] is False
    assert train.batch_size == 100
    assert train.drop_last is True
    assert train.sampler.__class__.__name__ == "RandomSampler"
    assert validation.batch_size == 200
    assert validation.drop_last is False
    assert test.sampler.__class__.__name__ == "SequentialSampler"


def test_other_dataset_split_is_seeded_and_cifar_augmentation_is_train_only():
    FakeVisionDataset.calls = []
    with _patch_dataset("CIFAR10"):
        first, first_val, _ = canonical_loaders.get_dataloaders(
            "cifar10", 4, val_split=0.25, seed=11, backend="local", download=False
        )
    with _patch_dataset("CIFAR10"):
        second, second_val, _ = canonical_loaders.get_dataloaders(
            "cifar10", 4, val_split=0.25, seed=11, backend="local", download=False
        )

    assert first_val is not None and second_val is not None
    assert first.dataset.subset.indices == second.dataset.subset.indices
    assert first_val.dataset.subset.indices == second_val.dataset.subset.indices
    assert len(get_transforms("cifar10", train=True).transforms) == 4
    assert len(get_transforms("cifar10", train=False).transforms) == 2
    assert first.dataset.transform is not first_val.dataset.transform


def test_loader_precedence_and_algorithm_batch_defaults():
    FakeVisionDataset.calls = []
    config = {
        "general": {"backend": "local"},
        "backend": {"local": {"data_loader": {"num_workers": 1, "pin_memory": True}}},
        "data_loader": {"num_workers": 4, "pin_memory": False},
    }
    with _patch_dataset("MNIST"):
        train, _, _ = canonical_loaders.get_dataloaders(
            "mnist",
            3,
            config=config,
            backend="local",
            num_workers=0,
            pin_memory=False,
            download=False,
        )
    assert train.num_workers == 0
    assert train.pin_memory is False
    assert train.persistent_workers is False

    with patch.object(
        canonical_loaders, "get_dataloaders", return_value=(1, 2, 3)
    ) as loader:
        result = build_dataloaders(
            {
                "algorithm": {"name": "FF"},
                "general": {"backend": "local", "seed": 2},
                "data": {"name": "MNIST", "download": False},
            }
        )
    assert result == (1, 2, 3)
    assert loader.call_args.kwargs["batch_size"] == 100
    assert loader.call_args.kwargs["download"] is False


def test_legacy_import_paths_are_retired_and_canonical_policy_works():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert not (root / "src" / "utils").exists()
    assert not (root / "src" / "data_utils").exists()

    assert (
        backend_policy.get_execution_backend({"general": {"backend": "local"}}).name
        == "local"
    )
