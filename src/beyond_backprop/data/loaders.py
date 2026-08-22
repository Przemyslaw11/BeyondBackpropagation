"""Typed boundary around the legacy loader while its split logic is migrated."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config.models import AlgorithmName, DatasetName, ExperimentConfig
from .registry import get_dataset_spec


def build_dataloaders(
    config: ExperimentConfig | Mapping[str, Any],
) -> tuple[Any, Any | None, Any]:
    """Build loaders with the canonical data policy and legacy split behavior.

    The implementation remains delegated to ``src.data_utils.datasets`` for
    this migration step. The adapter makes download policy, dataset registry,
    and configuration precedence explicit without duplicating split logic.
    """

    if isinstance(config, ExperimentConfig):
        resolved = config.to_mapping()
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
    from src.data_utils.datasets import get_dataloaders as legacy_get_dataloaders

    return legacy_get_dataloaders(
        dataset_name=dataset_name.value,
        batch_size=batch_size,
        data_root=data_root,
        val_split=val_split,
        seed=seed,
        config=resolved,
        backend=backend,
        download=download,
    )
