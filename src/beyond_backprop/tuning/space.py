"""Canonical algorithm-specific search spaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import exp, log
from random import Random
from typing import Any


@dataclass(frozen=True)
class SearchParameter:
    name: str
    target: str
    low: float
    high: float
    kind: str = "float"
    log_scale: bool = True

    def suggest(self, trial: Any) -> float | int:
        if self.kind == "int":
            return int(trial.suggest_int(self.name, int(self.low), int(self.high)))
        return float(
            trial.suggest_float(self.name, self.low, self.high, log=self.log_scale)
        )

    def sample(self, random: Random) -> float | int:
        if self.kind == "int":
            return random.randint(int(self.low), int(self.high))
        if self.log_scale:
            return exp(random.uniform(log(self.low), log(self.high)))
        return random.uniform(self.low, self.high)


def _range(
    config: dict[str, Any], key: str, fallback: tuple[float, float]
) -> tuple[float, float]:
    value = config.get(key, fallback)
    return float(value[0]), float(value[1])


def search_space(config: dict[str, Any]) -> tuple[SearchParameter, ...]:
    tuning = dict(config.get("tuning", {}))
    algorithm = str(config.get("algorithm", {}).get("name", "bp")).lower()
    if algorithm == "bp":
        parameters = [
            SearchParameter(
                "lr", "optimizer.lr", *_range(tuning, "lr_range", (1e-5, 1e-2))
            ),
            SearchParameter(
                "wd",
                "optimizer.weight_decay",
                *_range(tuning, "wd_range", (1e-6, 1e-3)),
            ),
        ]
        if str(config.get("optimizer", {}).get("type", "AdamW")).lower() == "sgd":
            parameters.append(
                SearchParameter(
                    "momentum",
                    "optimizer.momentum",
                    *_range(tuning, "momentum_range", (0.8, 0.99)),
                    log_scale=False,
                )
            )
        return tuple(parameters)
    if algorithm == "ff":
        return tuple(
            SearchParameter(name, target, *_range(tuning, key, fallback))
            for name, target, key, fallback in (
                (
                    "ff_lr",
                    "algorithm_params.ff_learning_rate",
                    "ff_lr_range",
                    (1e-5, 1e-2),
                ),
                (
                    "ff_wd",
                    "algorithm_params.ff_weight_decay",
                    "ff_wd_range",
                    (1e-6, 1e-3),
                ),
                (
                    "ds_lr",
                    "algorithm_params.downstream_learning_rate",
                    "ds_lr_range",
                    (1e-4, 1e-1),
                ),
                (
                    "ds_wd",
                    "algorithm_params.downstream_weight_decay",
                    "ds_wd_range",
                    (1e-5, 1e-2),
                ),
            )
        )
    if algorithm == "mf":
        low, high = _range(tuning, "mf_epochs_per_layer_range", (5, 50))
        return (
            SearchParameter(
                "lr",
                "algorithm_params.lr",
                *_range(tuning, "mf_lr_range", (1e-5, 1e-2)),
            ),
            SearchParameter(
                "epochs_per_layer",
                "algorithm_params.epochs_per_layer",
                low,
                high,
                kind="int",
                log_scale=False,
            ),
        )
    if algorithm == "cafo":
        fallback_lr = _range(tuning, "lr_range", (1e-5, 1e-2))
        fallback_wd = _range(tuning, "wd_range", (1e-6, 1e-3))
        epoch_low, epoch_high = _range(tuning, "cafo_epochs_per_block_range", (10, 200))
        parameters = [
            SearchParameter(
                "pred_lr",
                "algorithm_params.predictor_lr",
                *_range(tuning, "cafo_predictor_lr_range", fallback_lr),
            ),
            SearchParameter(
                "epochs_per_block",
                "algorithm_params.num_epochs_per_block",
                epoch_low,
                epoch_high,
                kind="int",
                log_scale=False,
            ),
            SearchParameter(
                "pred_wd",
                "algorithm_params.predictor_weight_decay",
                *_range(tuning, "cafo_predictor_wd_range", fallback_wd),
            ),
        ]
        if config.get("algorithm_params", {}).get("train_blocks", False):
            parameters.extend(
                (
                    SearchParameter(
                        "block_lr",
                        "algorithm_params.block_lr",
                        *_range(tuning, "cafo_block_lr_range", (1e-6, 1e-3)),
                    ),
                    SearchParameter(
                        "block_wd",
                        "algorithm_params.block_weight_decay",
                        *_range(tuning, "cafo_block_wd_range", (1e-7, 1e-4)),
                    ),
                )
            )
        return tuple(parameters)
    raise ValueError(f"Unsupported tuning algorithm: {algorithm}")


def generate_trial_config(
    base_config: dict[str, Any], params: dict[str, float | int], trial_number: int
) -> dict[str, Any]:
    """Apply suggested values and isolate trial randomness/infrastructure."""

    config = deepcopy(base_config)
    for parameter in search_space(config):
        if parameter.name not in params:
            raise ValueError(f"Missing suggested parameter: {parameter.name}")
        section, key = parameter.target.split(".", 1)
        config.setdefault(section, {})[key] = params[parameter.name]
    general = config.setdefault("general", {})
    general["seed"] = int(general.get("seed", 42)) + trial_number
    config.setdefault("monitoring", {}).update(
        {"enabled": False, "energy_enabled": False}
    )
    config.setdefault("tracking", {}).update({"enabled": False})
    config.setdefault("profiling", {}).update({"enabled": False})
    config.setdefault("carbon_tracker", {}).update({"enabled": False})
    config.setdefault("logging", {}).setdefault("wandb", {})["use_wandb"] = False
    return config


__all__ = ["SearchParameter", "generate_trial_config", "search_space"]
