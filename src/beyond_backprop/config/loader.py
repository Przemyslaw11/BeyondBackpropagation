"""Deterministic YAML loading, legacy normalization, and validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .models import (
    AlgorithmName,
    ArchitectureName,
    BackendName,
    DatasetName,
    ExperimentConfig,
)


class ConfigValidationError(ValueError):
    """Raised when a resolved configuration is not scientifically actionable."""


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigValidationError(
            f"Top-level YAML value must be a mapping: {config_path}"
        )
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; lists and scalars replace the base value."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def normalize_legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize known configuration defects without changing algorithm values.

    This function is deliberately explicit. It is the only compatibility point
    for historical spellings and misplaced fields, so a resolved config remains
    auditable.
    """

    normalized = copy.deepcopy(dict(config))
    data = normalized.setdefault("data", {})
    if "input_channels'" in data:
        if "input_channels" not in data:
            data["input_channels"] = data["input_channels'"]
        del data["input_channels'"]

    algorithm_params = normalized.setdefault("algorithm_params", {})
    for key in (
        "predictor_early_stopping_enabled",
        "predictor_early_stopping_metric",
        "predictor_early_stopping_patience",
        "predictor_early_stopping_mode",
        "predictor_early_stopping_min_delta",
    ):
        if key in normalized:
            algorithm_params.setdefault(key, normalized.pop(key))

    if "epochs_per_block" in algorithm_params:
        algorithm_params.setdefault(
            "num_epochs_per_block", algorithm_params["epochs_per_block"]
        )
        del algorithm_params["epochs_per_block"]

    return normalized


_ALLOWED_TOP_LEVEL = {
    "experiment_name",
    "general",
    "backend",
    "training",
    "data_loader",
    "data",
    "optimizer",
    "algorithm",
    "algorithm_params",
    "model",
    "logging",
    "monitoring",
    "carbon_tracker",
    "profiling",
    "checkpointing",
    "tuning",
    "tracking",
}
_ALLOWED_SECTION_KEYS = {
    "general": {"seed", "device", "backend"},
    "backend": {"local", "slurm"},
    "training": {
        "epochs",
        "criterion",
        "log_interval",
        "early_stopping_enabled",
        "early_stopping_metric",
        "early_stopping_patience",
        "early_stopping_mode",
        "early_stopping_min_delta",
    },
    "data_loader": {"batch_size", "num_workers", "pin_memory", "shuffle", "drop_last"},
    "data": {
        "root",
        "download",
        "val_split",
        "name",
        "num_classes",
        "input_channels",
        "image_size",
    },
    "optimizer": {"type", "lr", "weight_decay", "momentum", "betas"},
    "algorithm": {"name"},
    "model": {"name", "params"},
    "logging": {"level", "wandb", "log_file"},
    "monitoring": {"enabled", "energy_enabled", "energy_interval_sec"},
    "carbon_tracker": {"enabled", "mode", "output_dir", "country_iso_code"},
    "profiling": {"enabled", "verbose"},
    "checkpointing": {"checkpoint_dir", "save_best_metric"},
    "tuning": {
        "enabled",
        "n_trials",
        "direction",
        "metric",
        "sampler",
        "pruner",
        "lr_range",
        "wd_range",
        "ff_lr_range",
        "ff_wd_range",
        "ds_lr_range",
        "ds_wd_range",
        "mf_lr_range",
        "mf_epochs_per_layer_range",
        "num_epochs",
        "cafo_predictor_lr_range",
        "cafo_predictor_wd_range",
        "cafo_block_lr_range",
        "cafo_block_wd_range",
        "cafo_epochs_per_block_range",
    },
    "tracking": {"enabled", "project", "entity", "mode", "run_name"},
}
_ALLOWED_BACKEND_KEYS = {"results_dir", "log_file", "data_loader", "env"}
_ALLOWED_BACKEND_LOADER_KEYS = {"num_workers", "pin_memory"}
_ALLOWED_WANDB_KEYS = {"use_wandb", "project", "entity", "mode", "name"}
_ALLOWED_ALGORITHM_PARAMS = {
    "aggregation_method",
    "block_lr",
    "block_optimizer_type",
    "block_training_epochs",
    "block_weight_decay",
    "downstream_learning_rate",
    "downstream_momentum",
    "downstream_weight_decay",
    "epochs_per_block",
    "epochs_per_layer",
    "ff_learning_rate",
    "ff_momentum",
    "ff_weight_decay",
    "log_interval",
    "loss_type",
    "lr",
    "mf_early_stopping_enabled",
    "mf_early_stopping_min_delta",
    "mf_early_stopping_patience",
    "num_epochs_per_block",
    "optimizer_type",
    "peer_momentum",
    "peer_normalization_factor",
    "predictor_early_stopping_enabled",
    "predictor_early_stopping_metric",
    "predictor_early_stopping_min_delta",
    "predictor_early_stopping_mode",
    "predictor_early_stopping_patience",
    "predictor_lr",
    "predictor_optimizer_type",
    "predictor_weight_decay",
    "threshold",
    "train_blocks",
    "weight_decay",
}
_ALLOWED_MODEL_PARAMS = {
    "activation",
    "bias",
    "bias_init",
    "block_channels",
    "hidden_dims",
    "input_dim",
    "kernel_size",
    "norm_eps",
    "pool_kernel_size",
    "pool_stride",
    "use_batchnorm",
}


def _check_unknown_keys(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise ConfigValidationError(f"Unknown top-level configuration keys: {unknown}")
    for section, allowed in _ALLOWED_SECTION_KEYS.items():
        value = config.get(section, {})
        if not isinstance(value, Mapping):
            raise ConfigValidationError(
                f"Configuration section '{section}' must be a mapping"
            )
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigValidationError(f"Unknown keys in '{section}': {unknown}")

    backend = config.get("backend", {})
    for backend_name, backend_value in backend.items():
        if not isinstance(backend_value, Mapping):
            raise ConfigValidationError(
                f"Configuration section 'backend.{backend_name}' must be a mapping"
            )
        unknown = sorted(set(backend_value) - _ALLOWED_BACKEND_KEYS)
        if unknown:
            raise ConfigValidationError(
                f"Unknown keys in 'backend.{backend_name}': {unknown}"
            )
        loader = backend_value.get("data_loader", {})
        if not isinstance(loader, Mapping):
            raise ConfigValidationError(
                f"Configuration section 'backend.{backend_name}.data_loader' "
                "must be a mapping"
            )
        unknown = sorted(set(loader) - _ALLOWED_BACKEND_LOADER_KEYS)
        if unknown:
            raise ConfigValidationError(
                f"Unknown keys in 'backend.{backend_name}.data_loader': {unknown}"
            )

    logging_config = config.get("logging", {})
    wandb_config = logging_config.get("wandb", {})
    if not isinstance(wandb_config, Mapping):
        raise ConfigValidationError(
            "Configuration section 'logging.wandb' must be a mapping"
        )
    unknown = sorted(set(wandb_config) - _ALLOWED_WANDB_KEYS)
    if unknown:
        raise ConfigValidationError(f"Unknown keys in 'logging.wandb': {unknown}")

    algorithm_params = config.get("algorithm_params", {})
    if not isinstance(algorithm_params, Mapping):
        raise ConfigValidationError(
            "Configuration section 'algorithm_params' must be a mapping"
        )
    unknown = sorted(set(algorithm_params) - _ALLOWED_ALGORITHM_PARAMS)
    if unknown:
        raise ConfigValidationError(f"Unknown keys in 'algorithm_params': {unknown}")
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ConfigValidationError("Configuration section 'model' must be a mapping")
    params = model.get("params", {})
    if not isinstance(params, Mapping):
        raise ConfigValidationError(
            "Configuration section 'model.params' must be a mapping"
        )
    unknown = sorted(set(params) - _ALLOWED_MODEL_PARAMS)
    if unknown:
        raise ConfigValidationError(f"Unknown keys in 'model.params': {unknown}")


def validate_mapping(config: Mapping[str, Any]) -> None:
    _check_unknown_keys(config)
    algorithm = AlgorithmName.parse(config.get("algorithm", {}).get("name", "bp"))
    architecture = ArchitectureName.parse(config.get("model", {}).get("name", "mf_mlp"))
    dataset = DatasetName.parse(config.get("data", {}).get("name", "mnist"))
    if algorithm is AlgorithmName.FF and architecture is not ArchitectureName.FF_MLP:
        raise ConfigValidationError("FF requires model.name=FF_MLP")
    if algorithm is AlgorithmName.MF and architecture is not ArchitectureName.MF_MLP:
        raise ConfigValidationError("MF requires model.name=MF_MLP")
    if (
        algorithm is AlgorithmName.CAFO
        and architecture is not ArchitectureName.CAFO_CNN
    ):
        raise ConfigValidationError("CaFo requires model.name=CaFo_CNN")
    data = config.get("data", {})
    val_split = float(data.get("val_split", 0.1))
    if not 0.0 <= val_split < 1.0:
        raise ConfigValidationError("data.val_split must be in [0, 1)")
    loader = config.get("data_loader", {})
    if int(loader.get("batch_size", 128)) <= 0:
        raise ConfigValidationError("data_loader.batch_size must be positive")
    if int(loader.get("num_workers", 0)) < 0:
        raise ConfigValidationError("data_loader.num_workers must be non-negative")
    for field_name in ("num_classes", "input_channels", "image_size"):
        if field_name in data and int(data[field_name]) <= 0:
            raise ConfigValidationError(f"data.{field_name} must be positive")
    training = config.get("training", {})
    mode = str(training.get("early_stopping_mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ConfigValidationError(
            "training.early_stopping_mode must be 'min' or 'max'"
        )
    if int(training.get("epochs", 100)) <= 0:
        raise ConfigValidationError("training.epochs must be positive")
    if int(training.get("early_stopping_patience", 0)) < 0:
        raise ConfigValidationError(
            "training.early_stopping_patience must be non-negative"
        )
    if float(training.get("early_stopping_min_delta", 0.0)) < 0:
        raise ConfigValidationError(
            "training.early_stopping_min_delta must be non-negative"
        )
    metric = str(training.get("early_stopping_metric", "")).lower()
    if metric and ("acc" in metric or "accuracy" in metric) and mode != "max":
        raise ConfigValidationError(
            "Accuracy early-stopping metrics require training.early_stopping_mode=max"
        )
    if metric and "loss" in metric and mode != "min":
        raise ConfigValidationError(
            "Loss early-stopping metrics require training.early_stopping_mode=min"
        )
    if int(config.get("general", {}).get("seed", 42)) < 0:
        raise ConfigValidationError("general.seed must be non-negative")

    optimizer = config.get("optimizer", {})
    if float(optimizer.get("lr", 0.001)) <= 0:
        raise ConfigValidationError("optimizer.lr must be positive")
    if float(optimizer.get("weight_decay", 0.0)) < 0:
        raise ConfigValidationError("optimizer.weight_decay must be non-negative")

    model_params = config.get("model", {}).get("params", {})
    for field_name in ("hidden_dims", "block_channels"):
        if field_name in model_params:
            values = model_params[field_name]
            if not isinstance(values, (list, tuple)) or not values:
                raise ConfigValidationError(
                    f"model.params.{field_name} must be non-empty"
                )
            if any(int(value) <= 0 for value in values):
                raise ConfigValidationError(
                    f"model.params.{field_name} values must be positive"
                )

    tuning = config.get("tuning", {})
    if int(tuning.get("n_trials", 1)) <= 0:
        raise ConfigValidationError("tuning.n_trials must be positive")
    for key, value in tuning.items():
        if key.endswith("_range"):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ConfigValidationError(
                    f"tuning.{key} must contain exactly two values"
                )
            if float(value[0]) >= float(value[1]):
                raise ConfigValidationError(
                    f"tuning.{key} lower bound must be less than upper bound"
                )
    if not isinstance(dataset.value, str):  # pragma: no cover - enum invariant
        raise ConfigValidationError("Invalid dataset enum")


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_mapping(
    config_path: str | os.PathLike[str],
    base_config_path: str | os.PathLike[str] = "configs/base.yaml",
) -> dict[str, Any]:
    """Load, merge, normalize, and strictly validate a YAML configuration."""

    base = _load_yaml(base_config_path) if Path(base_config_path).exists() else {}
    specific = _load_yaml(config_path)
    merged = normalize_legacy_config(_deep_merge(base, specific))
    validate_mapping(merged)
    return merged


def load_experiment_config(
    config_path: str | os.PathLike[str],
    base_config_path: str | os.PathLike[str] = "configs/base.yaml",
) -> ExperimentConfig:
    resolved = load_mapping(config_path, base_config_path)
    return _config_from_mapping(resolved, config_path=config_path)


def _config_from_mapping(
    resolved: Mapping[str, Any],
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> ExperimentConfig:
    """Build the typed contract from an already validated mapping."""
    general = resolved.get("general", {})
    data = resolved.get("data", {})
    loader = resolved.get("data_loader", {})
    model = resolved.get("model", {})
    backend = BackendName.parse(general.get("backend", "slurm"))
    algorithm = AlgorithmName.parse(resolved.get("algorithm", {}).get("name", "bp"))
    architecture = ArchitectureName.parse(model.get("name", "mf_mlp"))
    dataset = DatasetName.parse(data.get("name", "mnist"))
    return ExperimentConfig(
        experiment_name=str(
            resolved.get(
                "experiment_name",
                Path(config_path).stem if config_path is not None else "experiment",
            )
        ),
        algorithm=algorithm,
        architecture=architecture,
        dataset=dataset,
        backend=backend,
        device=str(general.get("device", "auto")),
        seed=int(general.get("seed", 42)),
        batch_size=int(
            loader.get("batch_size", 100 if algorithm is AlgorithmName.FF else 128)
        ),
        num_workers=int(loader.get("num_workers", 0)),
        pin_memory=bool(loader.get("pin_memory", False)),
        data_root=str(data.get("root", "./data")),
        download=bool(data.get("download", True)),
        val_split=float(data.get("val_split", 0.1)),
        num_classes=int(data.get("num_classes", 10)),
        input_channels=int(data.get("input_channels", 1)),
        image_size=int(data.get("image_size", 28)),
        model_params=model.get("params", {}),
        algorithm_params=resolved.get("algorithm_params", {}),
        optimizer=resolved.get("optimizer", {}),
        training=resolved.get("training", {}),
        monitoring=resolved.get("monitoring", {}),
        tracking=resolved.get("tracking", resolved.get("logging", {}).get("wandb", {})),
        resolved=dict(resolved),
        config_hash=resolved_config_hash(resolved),
    )


def save_resolved_config(
    config: ExperimentConfig, path: str | os.PathLike[str]
) -> None:
    """Persist the exact resolved mapping used to construct the typed object."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_mapping(), handle, sort_keys=False)
