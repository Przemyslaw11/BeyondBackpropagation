"""Canonical preprocessing implementation moved from the legacy namespace."""

from __future__ import annotations

import logging

import torchvision.transforms as T  # noqa: N812 - PyTorch convention

logger = logging.getLogger(__name__)

DATASET_STATS: dict[str, dict[str, tuple[float, ...]]] = {
    "fashionmnist": {"mean": (0.2860,), "std": (0.3530,)},
    "mnist": {"mean": (0.1307,), "std": (0.3081,)},
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
    """Return the repository's train or evaluation transform policy."""

    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_STATS:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Available: {list(DATASET_STATS)}"
        )

    stats = DATASET_STATS[dataset_key]
    transform_list = []
    if train and dataset_key in {"cifar10", "cifar100"}:
        transform_list.extend(
            [
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
            ]
        )
    transform_list.extend([T.ToTensor(), T.Normalize(stats["mean"], stats["std"])])
    logger.debug(
        "Transforms for dataset '%s' (train=%s): %s",
        dataset_name,
        train,
        transform_list,
    )
    return T.Compose(transform_list)
