"""Functions and classes for creating PyTorch DataLoaders."""

import logging
from typing import Any, Callable, Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .preprocessing import get_transforms

logger = logging.getLogger(__name__)


class TransformedSubset(Dataset):
    """A Subset that applies a transform independently of the original dataset."""

    def __init__(self, subset: Subset, transform: Optional[Callable] = None) -> None:
        """Initializes the TransformedSubset.

        Args:
            subset: The original subset of data.
            transform: The transform to be applied to the data.
        """
        self.subset = subset
        self.transform = transform
        self.dataset = subset.dataset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Retrieves an item from the dataset and applies the transform."""
        data, target = self.dataset[self.subset.indices[index]]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self) -> int:
        """Returns the number of samples in the subset."""
        return len(self.subset.indices)


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    data_root: str = "./data",
    val_split: float = 0.1,
    seed: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Creates training, validation, and test DataLoaders for a specified dataset.

    Note:
        For the 'mnist' dataset, this function implements a specific 50k/10k
        fixed split, ignoring the `val_split` argument.

    Args:
        dataset_name: Name of the dataset (e.g., 'FashionMNIST', 'cifar10').
        batch_size: The number of samples per batch.
        data_root: The root directory for the dataset.
        val_split: The fraction of the training data to use for validation.
        seed: Random seed for reproducibility of the split.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning.
        download: If True, downloads the dataset if it is not found locally.

    Returns:
        A tuple containing the training, validation, and test DataLoaders.
    """
    dataset_name = dataset_name.lower()
    logger.info(f"Loading dataset: {dataset_name.upper()} from {data_root}")

    try:
        train_transform = get_transforms(dataset_name, train=True)
        test_transform = get_transforms(dataset_name, train=False)
    except ValueError as e:
        logger.error(f"Failed to get transforms: {e}")
        raise

    dataset_class = None
    if dataset_name == "fashionmnist":
        dataset_class = torchvision.datasets.FashionMNIST
    elif dataset_name == "mnist":
        dataset_class = torchvision.datasets.MNIST
    elif dataset_name == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        full_train_dataset_raw = dataset_class(
            root=data_root,
            train=True,
            download=download,
            transform=None,
        )
        test_dataset = dataset_class(
            root=data_root, train=False, download=download, transform=test_transform
        )
    except Exception as e:
        logger.error(
            f"Failed to load dataset {dataset_name.upper()}: {e}", exc_info=True
        )
        if isinstance(e, RuntimeError) and "download=True" in str(e) and not download:
            logger.error("Dataset not found locally and download=False.")
            raise FileNotFoundError(
                f"Dataset {dataset_name.upper()} not found in {data_root} and "
                "download is disabled."
            ) from e
        raise

    train_dataset: Dataset
    val_dataset: Optional[Dataset] = None

    if dataset_name == "mnist":
        logger.info("Applying specific MNIST fixed split: 50k train / 10k validation.")
        if len(full_train_dataset_raw) != 60000:
            logger.warning(
                f"MNIST training set size is not 60000 ({len(full_train_dataset_raw)})."
                " Fixed split might be incorrect."
            )
        train_indices = list(range(50000))
        val_indices = list(range(50000, 60000))

        if (
            not train_indices
            or not val_indices
            or val_indices[-1] >= len(full_train_dataset_raw)
        ):
            logger.error(
                "Failed to apply fixed 50k/10k split to MNIST. Check dataset integrity."
            )
            logger.warning("Falling back to random split due to fixed split error.")
            train_indices, val_indices = None, None
        else:
            train_subset = Subset(full_train_dataset_raw, train_indices)
            val_subset = Subset(full_train_dataset_raw, val_indices)
            train_dataset = TransformedSubset(train_subset, transform=train_transform)
            val_dataset = TransformedSubset(val_subset, transform=test_transform)
            logger.info(
                f"Created fixed MNIST split: {len(train_dataset)} train / "
                f"{len(val_dataset)} validation samples."
            )

    if "train_dataset" not in locals():
        if not 0.0 <= val_split < 1.0:
            raise ValueError(
                f"Validation split must be between 0.0 and 1.0 (exclusive of 1.0), got {val_split}"
            )

        if val_split > 0.0:
            num_train_total = len(full_train_dataset_raw)
            num_val = int(num_train_total * val_split)
            num_train_split = num_train_total - num_val
            if num_val == 0 or num_train_split == 0:
                logger.warning(
                    f"Validation split {val_split} resulted in {num_train_split} "
                    f"train / {num_val} validation samples for dataset size "
                    f"{num_train_total}. Using full dataset for training."
                )
                train_dataset = TransformedSubset(
                    Subset(full_train_dataset_raw, range(num_train_total)),
                    transform=train_transform,
                )
                val_dataset = None
            else:
                logger.info(
                    f"Splitting training data randomly: {num_train_split} train / "
                    f"{num_val} validation samples."
                )
                generator = (
                    torch.Generator().manual_seed(seed) if seed is not None else None
                )
                train_subset, val_subset = random_split(
                    full_train_dataset_raw,
                    [num_train_split, num_val],
                    generator=generator,
                )
                train_dataset = TransformedSubset(
                    train_subset, transform=train_transform
                )
                val_dataset = TransformedSubset(val_subset, transform=test_transform)
                logger.info(
                    "Applied training transforms to train subset and "
                    "test transforms to validation subset."
                )
        else:
            logger.info("Validation split is 0, using full training set for training.")
            train_dataset = TransformedSubset(
                Subset(full_train_dataset_raw, range(len(full_train_dataset_raw))),
                transform=train_transform,
            )
            val_dataset = None
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
        )
    elif val_split > 0.0 and dataset_name != "mnist":
        logger.warning(
            "Validation dataset is empty after random split. Validation loader will be None."
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    logger.info(
        f"DataLoaders created: "
        f"Train ({len(train_loader)} batches, {len(train_dataset)} samples), "
        f"Validation ({len(val_loader) if val_loader else 0} batches, "
        f"{len(val_dataset) if val_dataset else 0} samples), "
        f"Test ({len(test_loader)} batches, {len(test_dataset)} samples)"
    )

    return train_loader, val_loader, test_loader
