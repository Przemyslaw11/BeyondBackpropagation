"""Image transformation and normalization utilities for datasets."""

import logging

import torchvision.transforms as T

logger = logging.getLogger(__name__)

# --- Normalization Constants ---
DATASET_STATS = {
    "fashionmnist": {
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


def get_transforms(dataset_name: str, train: bool = True) -> T.Compose:
    """Returns the appropriate torchvision transforms for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'FashionMNIST', 'cifar10').
        train: If True, applies training augmentations. Otherwise, returns
            transforms for validation/testing.

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

    if train:
        if dataset_key in ["cifar10", "cifar100"]:
            transform_list.extend(
                [
                    T.RandomCrop(32, padding=4, padding_mode="reflect"),
                    T.RandomHorizontalFlip(),
                ]
            )

    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean, std))

    logger.debug(
        f"Transforms for dataset '{dataset_name}' (train={train}): {transform_list}"
    )
    return T.Compose(transform_list)
