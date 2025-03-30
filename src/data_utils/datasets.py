# File: src/data_utils/datasets.py
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split, Dataset  # Import Dataset
import os
import logging
from typing import Tuple, Optional, Callable  # Added Callable

# Import transforms from the sibling module
from .preprocessing import get_transforms

logger = logging.getLogger(__name__)


# --- Custom Subset class to handle transform application ---
# This is crucial because the original dataset instance (full_train_dataset)
# has the *training* transforms. We need validation subset to use *test* transforms.
class TransformedSubset(Dataset):
    """
    A Subset that applies a specific transform independent of the original dataset's transform.
    """

    def __init__(self, subset: Subset, transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Get the *original* data item from the underlying dataset
        data, target = self.subset.dataset[self.subset.indices[index]]
        # Apply the specific transform provided to this subset
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset)


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    data_root: str = "./data",
    val_split: float = 0.1,
    seed: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:  # val_loader can be None
    """
    Creates training, validation, and test DataLoaders for a specified dataset.

    Args:
        dataset_name: Name of the dataset ('FashionMNIST', 'CIFAR10', 'CIFAR100').
        batch_size: Number of samples per batch.
        data_root: Root directory where the dataset will be stored/downloaded.
        val_split: Fraction of the training data to use for validation (e.g., 0.1 for 10%).
                   If 0, val_loader will be None.
        seed: Optional random seed for the train/validation split for reproducibility.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
                    before returning them (recommended for GPU training).
        download: If True, downloads the dataset if it's not found in data_root.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader). val_loader is None if val_split is 0.

    Raises:
        ValueError: If the dataset_name is not recognized or val_split is invalid.
        FileNotFoundError: If download is False and the dataset is not found.
    """
    dataset_name = dataset_name.lower()
    logger.info(f"Loading dataset: {dataset_name.upper()} from {data_root}")

    if not 0.0 <= val_split < 1.0:
        raise ValueError(
            f"Validation split must be between 0.0 and 1.0 (exclusive of 1.0), got {val_split}"
        )

    # --- Get Transforms ---
    try:
        # Get transforms *before* loading dataset - needed for validation split handling
        train_transform = get_transforms(dataset_name, train=True)
        test_transform = get_transforms(
            dataset_name, train=False
        )  # Use test transforms for val/test
    except ValueError as e:
        logger.error(f"Failed to get transforms: {e}")
        raise

    # --- Load Datasets ---
    dataset_class = None
    if dataset_name == "fashionmnist":
        dataset_class = torchvision.datasets.FashionMNIST
    elif dataset_name == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        # Load raw training data *without* transforms initially if splitting needed
        # If val_split > 0, we apply transforms later via TransformedSubset
        # If val_split == 0, we can apply train_transform directly here
        initial_train_transform = train_transform if val_split == 0.0 else None
        full_train_dataset = dataset_class(
            root=data_root,
            train=True,
            download=download,
            transform=initial_train_transform,  # Apply later if splitting
        )

        # Load test data with test transforms
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
                f"Dataset {dataset_name.upper()} not found in {data_root} and download is disabled."
            )
        raise  # Re-raise other errors

    # --- Split Train/Validation ---
    train_dataset: Dataset  # Type hint
    val_dataset: Optional[Dataset] = None  # val_dataset can be None

    if val_split > 0.0:
        num_train_total = len(full_train_dataset)
        num_val = int(num_train_total * val_split)
        num_train_split = num_train_total - num_val

        if num_val == 0:
            logger.warning(
                f"Validation split {val_split} resulted in 0 samples for dataset size {num_train_total}. Validation loader will be None."
            )
            # Use full dataset for training, apply train transforms
            train_dataset = TransformedSubset(
                full_train_dataset, transform=train_transform
            )
            val_dataset = None
        else:
            logger.info(
                f"Splitting training data: {num_train_split} train / {num_val} validation samples."
            )
            generator = (
                torch.Generator().manual_seed(seed) if seed is not None else None
            )
            # Split the *indices*
            train_subset, val_subset = random_split(
                full_train_dataset, [num_train_split, num_val], generator=generator
            )

            # Apply correct transforms using the wrapper class
            train_dataset = TransformedSubset(train_subset, transform=train_transform)
            val_dataset = TransformedSubset(val_subset, transform=test_transform)
            logger.info(
                "Applied training transforms to train subset and test transforms to validation subset."
            )
    else:
        # No validation split needed, use the already transformed full_train_dataset
        logger.info("Validation split is 0, using full training set for training.")
        train_dataset = full_train_dataset  # Already has train_transform applied
        val_dataset = None

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Often beneficial for training stability
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,  # Can use larger batch size for validation if memory allows
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    elif val_split > 0.0:
        # Log warning only if user intended a validation set but it ended up empty
        logger.warning("Validation dataset is empty. Validation loader will be None.")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # Can use larger batch size for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"DataLoaders created: Train ({len(train_loader)} batches), Validation ({len(val_loader) if val_loader else 0} batches), Test ({len(test_loader)} batches)"
    )

    return train_loader, val_loader, test_loader

