"""Canonical dataset construction and DataLoader policy.

This module owns the repository's data protocol.  The legacy
``src.data_utils.datasets`` import is an alias to this module, so callers can
be migrated without creating a second implementation of the split or loader
semantics.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..config.models import AlgorithmName, DatasetName, ExperimentConfig
from ..runtime.backend_policy import get_execution_backend
from .preprocessing import get_transforms
from .registry import get_dataset_spec

logger = logging.getLogger(__name__)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the per-worker PyTorch seed.

    PyTorch assigns each worker a deterministic seed when the DataLoader's
    generator is seeded.  Propagating that seed to the other common RNGs keeps
    augmentations and user datasets reproducible as well.
    """

    del worker_id  # The worker-specific value is already included in the torch seed.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class TransformedSubset(Dataset[Any]):
    """A subset that applies a transform independently of its source dataset."""

    def __init__(self, subset: Subset[Any], transform: Callable | None = None) -> None:
        self.subset = subset
        self.transform = transform
        self.dataset = subset.dataset

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data, target = self.dataset[self.subset.indices[index]]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self) -> int:
        return len(self.subset.indices)


def _dataset_class(dataset_name: DatasetName) -> type[Dataset[Any]]:
    spec = get_dataset_spec(dataset_name)
    return getattr(torchvision.datasets, spec.torchvision_name)


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    data_root: str = "./data",
    val_split: float = 0.1,
    seed: int | None = None,
    config: Mapping[str, Any] | None = None,
    backend: str = "slurm",
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    download: bool = True,
) -> tuple[DataLoader[Any], DataLoader[Any] | None, DataLoader[Any]]:
    """Create train, validation, and test loaders using the paper protocol.

    MNIST keeps its historical fixed 50k/10k split.  Other datasets use a
    seeded random split, and all validation/test subsets receive evaluation
    transforms even though they share the raw training dataset object.
    """
    dataset_key = DatasetName.parse(dataset_name)
    dataset_name_lower = dataset_key.value
    logger.info("Loading dataset: %s from %s", dataset_name_lower.upper(), data_root)

    train_transform = get_transforms(dataset_name_lower, train=True)
    evaluation_transform = get_transforms(dataset_name_lower, train=False)

    backend_policy = get_execution_backend({"general": {"backend": backend}})
    loader_defaults = backend_policy.resolve_dataloader_defaults(
        dict(config) if isinstance(config, Mapping) else None
    )
    resolved_num_workers = (
        int(num_workers)
        if num_workers is not None
        else int(loader_defaults["num_workers"])
    )
    resolved_pin_memory = (
        bool(pin_memory)
        if pin_memory is not None
        else bool(loader_defaults["pin_memory"])
    )
    if resolved_num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset_class = _dataset_class(dataset_key)
    try:
        full_train_dataset_raw = dataset_class(
            root=data_root,
            train=True,
            download=download,
            transform=None,
        )
        test_dataset = dataset_class(
            root=data_root,
            train=False,
            download=download,
            transform=evaluation_transform,
        )
    except Exception as exc:
        logger.error("Failed to load dataset %s: %s", dataset_name_lower.upper(), exc)
        if (
            isinstance(exc, RuntimeError)
            and "download=True" in str(exc)
            and not download
        ):
            raise FileNotFoundError(
                f"Dataset {dataset_name_lower.upper()} not found in {data_root} and "
                "download is disabled."
            ) from exc
        raise

    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any] | None = None

    # The fixed split is intentionally retained exactly for MNIST.  Synthetic
    # or incomplete fixtures fall through to the legacy random-split safety net.
    if dataset_key is DatasetName.MNIST:
        logger.info("Applying specific MNIST fixed split: 50k train / 10k validation.")
        train_indices = list(range(50000))
        val_indices = list(range(50000, 60000))
        if (
            not train_indices
            or not val_indices
            or val_indices[-1] >= len(full_train_dataset_raw)
        ):
            logger.warning(
                "MNIST fixed split unavailable for dataset size %d; using random split.",
                len(full_train_dataset_raw),
            )
        else:
            train_dataset = TransformedSubset(
                Subset(full_train_dataset_raw, train_indices), train_transform
            )
            val_dataset = TransformedSubset(
                Subset(full_train_dataset_raw, val_indices), evaluation_transform
            )

    if "train_dataset" not in locals():
        if not 0.0 <= val_split < 1.0:
            raise ValueError(
                "Validation split must be between 0.0 and 1.0 "
                f"(exclusive of 1.0), got {val_split}"
            )

        total_samples = len(full_train_dataset_raw)
        num_val = int(total_samples * val_split)
        num_train = total_samples - num_val
        if val_split > 0.0 and num_val > 0 and num_train > 0:
            generator = (
                torch.Generator().manual_seed(seed) if seed is not None else None
            )
            train_subset, val_subset = random_split(
                full_train_dataset_raw, [num_train, num_val], generator=generator
            )
            train_dataset = TransformedSubset(train_subset, train_transform)
            val_dataset = TransformedSubset(val_subset, evaluation_transform)
        else:
            if val_split > 0.0:
                logger.warning(
                    "Validation split %s produced %d train / %d validation samples; "
                    "using the full dataset for training.",
                    val_split,
                    num_train,
                    num_val,
                )
            train_dataset = TransformedSubset(
                Subset(full_train_dataset_raw, range(total_samples)), train_transform
            )

    persistent_workers = resolved_num_workers > 0
    loader_config = (
        dict(config.get("data_loader", {})) if isinstance(config, Mapping) else {}
    )
    train_shuffle = loader_config.get("shuffle", True)
    train_drop_last = loader_config.get("drop_last", True)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    common_loader_kwargs = {
        "num_workers": resolved_num_workers,
        "pin_memory": resolved_pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker if resolved_num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(train_shuffle),
        drop_last=bool(train_drop_last),
        generator=generator,
        **common_loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def build_dataloaders(
    config: ExperimentConfig | Mapping[str, Any],
) -> tuple[DataLoader[Any], DataLoader[Any] | None, DataLoader[Any]]:
    """Build loaders from a typed or legacy-compatible experiment mapping."""
    if isinstance(config, ExperimentConfig):
        resolved = config.to_mapping()
        algorithm = config.algorithm
        dataset_name = config.dataset
        batch_size = config.batch_size
        data_root = config.data_root
        val_split = config.val_split
        seed = config.seed
        backend = config.backend.value
        download = config.download
    else:
        resolved = dict(config)
        data = resolved.get("data", {})
        loader = resolved.get("data_loader", {})
        general = resolved.get("general", {})
        dataset_name = DatasetName.parse(data.get("name", "mnist"))
        algorithm = AlgorithmName.parse(resolved.get("algorithm", {}).get("name", "bp"))
        batch_size = int(
            loader.get("batch_size", 100 if algorithm is AlgorithmName.FF else 128)
        )
        data_root = str(data.get("root", "./data"))
        val_split = float(data.get("val_split", 0.1))
        seed = general.get("seed")
        backend = str(general.get("backend", "slurm"))
        download = bool(data.get("download", True))

    get_dataset_spec(dataset_name)
    return get_dataloaders(
        dataset_name=dataset_name.value,
        batch_size=batch_size,
        data_root=data_root,
        val_split=val_split,
        seed=seed,
        config=resolved,
        backend=backend,
        download=download,
    )


__all__ = ["TransformedSubset", "build_dataloaders", "get_dataloaders", "seed_worker"]
