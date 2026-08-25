"""Deterministic YAML loading, legacy normalization, and validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from numbers import Real
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
    if not isinstance(data, dict):
        raise ConfigValidationError("Configuration section 'data' must be a mapping")
    if "input_channels'" in data:
        if "input_channels" not in data:
            data["input_channels"] = data["input_channels'"]
        del data["input_channels'"]

    algorithm_params = normalized.setdefault("algorithm_params", {})
    if not isinstance(algorithm_params, dict):
        raise ConfigValidationError(
            "Configuration section 'algorithm_params' must be a mapping"
        )
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
    "results",
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
        "scheduler",
        "scheduler_params",
    },
    "data_loader": {
        "batch_size",
        "num_workers",
        "pin_memory",
        "shuffle",
        "drop_last",
        "persistent_workers",
    },
    "data": {
        "root",
        "download",
        "val_split",
        "name",
        "num_classes",
        "input_channels",
        "image_size",
    },
    "optimizer": {"type", "lr", "weight_decay", "momentum", "betas", "params"},
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
        "momentum_range",
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
    "results": {"dir"},
}
_ALLOWED_BACKEND_KEYS = {"results_dir", "log_file", "data_loader", "env"}
_ALLOWED_BACKEND_LOADER_KEYS = {"num_workers", "pin_memory"}
_ALLOWED_WANDB_KEYS = {"use_wandb", "project", "entity", "mode", "name"}
_ALGO_PARAMS_BP: frozenset[str] = frozenset()
_ALGO_PARAMS_FF = frozenset(
    {
        "downstream_learning_rate",
        "downstream_momentum",
        "downstream_weight_decay",
        "ff_learning_rate",
        "ff_momentum",
        "ff_weight_decay",
        "optimizer_type",
        "peer_momentum",
        "peer_normalization_factor",
        # Legacy: present in tracked FF/tuning configs but consumed by no trainer.
        "threshold",
    }
)
_ALGO_PARAMS_MF = frozenset(
    {
        "epochs_per_layer",
        "log_interval",
        "lr",
        "mf_early_stopping_enabled",
        "mf_early_stopping_min_delta",
        "mf_early_stopping_patience",
        "optimizer_type",
        "weight_decay",
    }
)
_ALGO_PARAMS_CAFO = frozenset(
    {
        "aggregation_method",
        "block_lr",
        "block_optimizer_type",
        "block_training_epochs",
        "block_weight_decay",
        "dfa_feedback_matrix_type",
        # Legacy alias normalized to num_epochs_per_block by
        # normalize_legacy_config; tolerated here for raw-mapping validation.
        "epochs_per_block",
        "log_interval",
        "loss_type",
        # Legacy: present in tracked CaFo configs but consumed by no trainer
        # (superseded by block_/predictor_ variants).
        "lr",
        "num_epochs_per_block",
        "optimizer_params",
        "optimizer_type",
        "predictor_early_stopping_enabled",
        "predictor_early_stopping_metric",
        "predictor_early_stopping_min_delta",
        "predictor_early_stopping_mode",
        "predictor_early_stopping_patience",
        "predictor_lr",
        "predictor_optimizer_type",
        "predictor_weight_decay",
        "train_blocks",
        "weight_decay",
    }
)
_ALLOWED_ALGORITHM_PARAMS_BY_ALGORITHM = {
    "bp": _ALGO_PARAMS_BP,
    "cafo": _ALGO_PARAMS_CAFO,
    "ff": _ALGO_PARAMS_FF,
    "mf": _ALGO_PARAMS_MF,
}
_ALL_ALGORITHM_PARAMS = (
    _ALGO_PARAMS_BP | _ALGO_PARAMS_CAFO | _ALGO_PARAMS_FF | _ALGO_PARAMS_MF
)
# Keys contributed to every experiment by configs/base.yaml during merge;
# they must be tolerated regardless of the resolved algorithm.
_BASE_TEMPLATE_PARAMS = frozenset(
    {
        "mf_early_stopping_enabled",
        "mf_early_stopping_min_delta",
        "mf_early_stopping_patience",
    }
)
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
    algorithm_section = config.get("algorithm")
    algorithm_name = ""
    if isinstance(algorithm_section, Mapping):
        name_value = algorithm_section.get("name")
        if isinstance(name_value, str):
            algorithm_name = name_value.strip().lower()
    # CFG-004: reject unknown algorithm_params keys for the resolved
    # algorithm. Base templates without an algorithm.name fall back to the
    # union so only truly unknown keys are rejected.
    algo_allowed: frozenset[str] = (
        _ALLOWED_ALGORITHM_PARAMS_BY_ALGORITHM.get(
            algorithm_name, _ALL_ALGORITHM_PARAMS
        )
        | _BASE_TEMPLATE_PARAMS
    )
    unknown = sorted(set(algorithm_params) - algo_allowed)
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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require(value: Any, expected: type | tuple[type, ...], path: str) -> None:
    if not isinstance(value, expected):
        names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ConfigValidationError(
            f"{path} must be of type {names}; got {type(value).__name__}"
        )


def _require_int(value: Any, path: str) -> None:
    if not _is_int(value):
        raise ConfigValidationError(
            f"{path} must be an integer; got {type(value).__name__}"
        )


def _require_number(value: Any, path: str) -> None:
    if not _is_number(value):
        raise ConfigValidationError(
            f"{path} must be numeric; got {type(value).__name__}"
        )


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"{path} must be a boolean; got {type(value).__name__}"
        )


def _validate_scalar_types(config: Mapping[str, Any]) -> None:
    """Reject YAML strings such as ``"false"`` instead of silently coercing them."""

    string_fields = {
        "experiment_name": config.get("experiment_name"),
    }
    for name, value in string_fields.items():
        if value is not None:
            _require(value, str, name)

    section_fields: dict[str, dict[str, tuple[type, ...] | type]] = {
        "general": {"seed": int, "device": str, "backend": str},
        "data_loader": {
            "batch_size": int,
            "num_workers": int,
            "pin_memory": bool,
            "shuffle": bool,
            "drop_last": bool,
            "persistent_workers": bool,
        },
        "data": {
            "root": str,
            "download": bool,
            "val_split": (int, float),
            "name": str,
            "num_classes": int,
            "input_channels": int,
            "image_size": int,
        },
        "algorithm": {"name": str},
        "model": {"name": str},
        "training": {
            "epochs": int,
            "criterion": str,
            "log_interval": int,
            "early_stopping_enabled": bool,
            "early_stopping_metric": str,
            "early_stopping_patience": int,
            "early_stopping_mode": str,
            "early_stopping_min_delta": (int, float),
            "scheduler": (str, type(None)),
        },
        "optimizer": {
            "type": str,
            "lr": (int, float),
            "weight_decay": (int, float),
            "momentum": (int, float),
        },
        "monitoring": {
            "enabled": bool,
            "energy_enabled": bool,
            "energy_interval_sec": (int, float),
        },
        "carbon_tracker": {
            "enabled": bool,
            "mode": str,
            "output_dir": str,
            "country_iso_code": str,
        },
        "profiling": {"enabled": bool, "verbose": bool},
        "checkpointing": {"checkpoint_dir": (str, type(None)), "save_best_metric": str},
        "tracking": {
            "enabled": bool,
            "project": str,
            "entity": str,
            "mode": str,
            "run_name": str,
        },
        "results": {"dir": str},
    }
    for section, fields in section_fields.items():
        value = config.get(section, {})
        if not isinstance(value, Mapping):
            continue
        for field_name, expected in fields.items():
            if field_name not in value:
                continue
            item = value[field_name]
            if expected is int:
                _require_int(item, f"{section}.{field_name}")
            elif expected is bool:
                _require_bool(item, f"{section}.{field_name}")
            elif expected == (int, float):
                _require_number(item, f"{section}.{field_name}")
            else:
                _require(item, expected, f"{section}.{field_name}")

    for section_name, value in (
        ("backend", config.get("backend", {})),
        ("logging", config.get("logging", {})),
    ):
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            if section_name == "backend":
                if not isinstance(item, Mapping):
                    continue
                for nested_key in ("results_dir", "log_file"):
                    nested = item.get(nested_key)
                    if nested is not None:
                        _require(nested, str, f"backend.{key}.{nested_key}")
            elif key in {"level", "log_file"} and item is not None:
                _require(item, str, f"logging.{key}")
        wandb = value.get("wandb", {})
        if isinstance(wandb, Mapping):
            for key in ("use_wandb",):
                if key in wandb:
                    _require_bool(wandb[key], f"logging.wandb.{key}")
            for key in ("project", "entity", "mode", "name"):
                if key in wandb and wandb[key] is not None:
                    _require(wandb[key], str, f"logging.wandb.{key}")

    model_params = config.get("model", {}).get("params", {})
    if isinstance(model_params, Mapping):
        for key in ("hidden_dims", "block_channels"):
            if key in model_params:
                values = model_params[key]
                if not isinstance(values, (list, tuple)):
                    raise ConfigValidationError(f"model.params.{key} must be a list")
                for index, item in enumerate(values):
                    _require_int(item, f"model.params.{key}[{index}]")
        for key in ("bias", "use_batchnorm"):
            if key in model_params:
                _require_bool(model_params[key], f"model.params.{key}")
        for key in ("input_dim", "kernel_size", "pool_kernel_size", "pool_stride"):
            if key in model_params:
                _require_int(model_params[key], f"model.params.{key}")
        for key in ("activation", "bias_init", "norm_eps"):
            if key in model_params and key == "activation":
                _require(model_params[key], str, f"model.params.{key}")
            elif key in model_params:
                _require_number(model_params[key], f"model.params.{key}")

    optimizer = config.get("optimizer", {})
    if isinstance(optimizer, Mapping) and "betas" in optimizer:
        betas = optimizer["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ConfigValidationError("optimizer.betas must be a two-item list")
        for index, item in enumerate(betas):
            _require_number(item, f"optimizer.betas[{index}]")

    tuning = config.get("tuning", {})
    if isinstance(tuning, Mapping):
        if "enabled" in tuning:
            _require_bool(tuning["enabled"], "tuning.enabled")
        if "n_trials" in tuning:
            _require_int(tuning["n_trials"], "tuning.n_trials")
        for key in ("direction", "metric", "sampler", "pruner"):
            if key in tuning:
                _require(tuning[key], str, f"tuning.{key}")
        for key, value in tuning.items():
            if key.endswith("_range"):
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                for index, item in enumerate(value):
                    _require_number(item, f"tuning.{key}[{index}]")


def validate_mapping(config: Mapping[str, Any]) -> None:
    _check_unknown_keys(config)
    _validate_scalar_types(config)
    try:
        algorithm = AlgorithmName.parse(config.get("algorithm", {}).get("name", "bp"))
        architecture = ArchitectureName.parse(
            config.get("model", {}).get("name", "mf_mlp")
        )
        DatasetName.parse(config.get("data", {}).get("name", "mnist"))
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(str(exc)) from exc
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
    val_split = data.get("val_split", 0.1)
    _require_number(val_split, "data.val_split")
    if not 0.0 <= val_split < 1.0:
        raise ConfigValidationError("data.val_split must be in [0, 1)")
    loader = config.get("data_loader", {})
    batch_size = loader.get("batch_size", 128)
    _require_int(batch_size, "data_loader.batch_size")
    if batch_size <= 0:
        raise ConfigValidationError("data_loader.batch_size must be positive")
    num_workers = loader.get("num_workers", 0)
    _require_int(num_workers, "data_loader.num_workers")
    if num_workers < 0:
        raise ConfigValidationError("data_loader.num_workers must be non-negative")
    for field_name in ("num_classes", "input_channels", "image_size"):
        if field_name in data:
            _require_int(data[field_name], f"data.{field_name}")
        if field_name in data and data[field_name] <= 0:
            raise ConfigValidationError(f"data.{field_name} must be positive")
    training = config.get("training", {})
    mode = str(training.get("early_stopping_mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ConfigValidationError(
            "training.early_stopping_mode must be 'min' or 'max'"
        )
    epochs = training.get("epochs", 100)
    _require_int(epochs, "training.epochs")
    if epochs <= 0:
        raise ConfigValidationError("training.epochs must be positive")
    patience = training.get("early_stopping_patience", 0)
    _require_int(patience, "training.early_stopping_patience")
    if patience < 0:
        raise ConfigValidationError(
            "training.early_stopping_patience must be non-negative"
        )
    min_delta = training.get("early_stopping_min_delta", 0.0)
    _require_number(min_delta, "training.early_stopping_min_delta")
    if min_delta < 0:
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
    seed = config.get("general", {}).get("seed", 42)
    _require_int(seed, "general.seed")
    if seed < 0:
        raise ConfigValidationError("general.seed must be non-negative")

    optimizer = config.get("optimizer", {})
    learning_rate = optimizer.get("lr", 0.001)
    _require_number(learning_rate, "optimizer.lr")
    if learning_rate <= 0:
        raise ConfigValidationError("optimizer.lr must be positive")
    weight_decay = optimizer.get("weight_decay", 0.0)
    _require_number(weight_decay, "optimizer.weight_decay")
    if weight_decay < 0:
        raise ConfigValidationError("optimizer.weight_decay must be non-negative")

    model_params = config.get("model", {}).get("params", {})
    for field_name in ("hidden_dims", "block_channels"):
        if field_name in model_params:
            values = model_params[field_name]
            if not isinstance(values, (list, tuple)) or not values:
                raise ConfigValidationError(
                    f"model.params.{field_name} must be non-empty"
                )
            if any(value <= 0 for value in values):
                raise ConfigValidationError(
                    f"model.params.{field_name} values must be positive"
                )

    tuning = config.get("tuning", {})
    n_trials = tuning.get("n_trials", 1)
    _require_int(n_trials, "tuning.n_trials")
    if n_trials <= 0:
        raise ConfigValidationError("tuning.n_trials must be positive")
    for key, value in tuning.items():
        if key.endswith("_range"):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ConfigValidationError(
                    f"tuning.{key} must contain exactly two values"
                )
            if value[0] >= value[1]:
                raise ConfigValidationError(
                    f"tuning.{key} lower bound must be less than upper bound"
                )
    tuning_direction = str(tuning.get("direction", "maximize")).lower()
    if tuning_direction not in {"maximize", "minimize"}:
        raise ConfigValidationError("tuning.direction must be 'maximize' or 'minimize'")


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_overrides(
    config: Mapping[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    """Apply explicit ``section.key=value`` overrides to a config copy."""

    result = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ConfigValidationError(
                f"Invalid override {override!r}; expected section.key=value"
            )
        path_text, value_text = override.split("=", 1)
        path = [part.strip() for part in path_text.split(".") if part.strip()]
        if not path:
            raise ConfigValidationError(
                f"Invalid override {override!r}; key path is empty"
            )
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(
                f"Invalid YAML value in override {override!r}: {exc}"
            ) from exc
        target: dict[str, Any] = result
        for key in path[:-1]:
            nested = target.get(key)
            if not isinstance(nested, dict):
                raise ConfigValidationError(
                    f"Cannot apply override {override!r}; '{key}' is not a mapping"
                )
            target = nested
        target[path[-1]] = value
    return result


def load_mapping(
    config_path: str | os.PathLike[str],
    base_config_path: str | os.PathLike[str] = "configs/base.yaml",
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Load, merge, normalize, and strictly validate a YAML configuration."""

    base = _load_yaml(base_config_path) if Path(base_config_path).exists() else {}
    specific = _load_yaml(config_path)
    merged = _deep_merge(base, specific)
    merged = apply_overrides(merged, overrides)
    merged = normalize_legacy_config(merged)
    validate_mapping(merged)
    return merged


def load_experiment_config(
    config_path: str | os.PathLike[str],
    base_config_path: str | os.PathLike[str] = "configs/base.yaml",
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    resolved = load_mapping(config_path, base_config_path, overrides)
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


def experiment_config_from_mapping(
    resolved: Mapping[str, Any],
) -> ExperimentConfig:
    """Build a typed config from a validated in-memory mapping."""
    validate_mapping(resolved)
    return _config_from_mapping(resolved)


def save_resolved_config(
    config: ExperimentConfig, path: str | os.PathLike[str]
) -> None:
    """Persist the exact resolved mapping used to construct the typed object."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_mapping(), handle, sort_keys=False)
