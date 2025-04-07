# File: src/data_utils/preprocessing.py
import torch
import torchvision.transforms as T  # Use T alias for brevity
import torchvision.transforms.functional as TF
from typing import Tuple, Dict, List  # Add List
import logging

logger = logging.getLogger(__name__)

# --- Normalization Constants ---
# Use more descriptive names and store in a dictionary for easier access
DATASET_STATS = {
    "fashionmnist": {
        # Values calculated on the standard training set (60k images)
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "mnist": { # <<< ADDED MNIST STATS >>>
        # Standard values for MNIST
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),  # Standard values
        "std": (0.2023, 0.1994, 0.2010),  # Standard values (alt: 0.247, 0.243, 0.261)
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),  # Standard values
        "std": (0.2675, 0.2565, 0.2761),  # Standard values
    },
    # Add other datasets if needed
}


def get_transforms(dataset_name: str, train: bool = True) -> T.Compose:
    """
    Returns the appropriate torchvision transforms for a given dataset and split.

    Args:
        dataset_name: Name of the dataset (case-insensitive, e.g., 'FashionMNIST', 'cifar10').
        train: Boolean indicating if the transforms are for the training set (True)
               or validation/test set (False). Augmentation is applied only if train is True.

    Returns:
        A torchvision.transforms.Compose object.

    Raises:
        ValueError: If the dataset_name is not recognized.
    """
    dataset_key = dataset_name.lower()

    if dataset_key not in DATASET_STATS:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Available: {list(DATASET_STATS.keys())}"
        )

    stats = DATASET_STATS[dataset_key]
    mean = stats["mean"]
    std = stats["std"]

    transform_list = []

    # --- Training specific transforms ---
    if train:
        if dataset_key in ["cifar10", "cifar100"]:
            transform_list.extend(
                [
                    T.RandomCrop(32, padding=4, padding_mode="reflect"),  # Pad and crop
                    T.RandomHorizontalFlip(),
                    # Add more augmentations if needed (e.g., T.RandAugment(), T.AutoAugment())
                ]
            )
        # No standard augmentation for MNIST/FashionMNIST in this setup

    # --- Common transforms (applied after training-specific ones) ---
    transform_list.append(
        T.ToTensor()
    )  # Converts PIL Image or numpy.ndarray to tensor and scales to [0, 1]
    transform_list.append(T.Normalize(mean, std))  # Normalize using dataset stats

    logger.debug(
        f"Transforms for dataset '{dataset_name}' (train={train}): {transform_list}"
    )
    return T.Compose(transform_list)