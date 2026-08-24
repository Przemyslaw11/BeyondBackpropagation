"""Canonical tuning execution with optional Optuna storage."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from random import Random
from typing import Any

from ..config import ExperimentConfig, experiment_config_from_mapping
from ..training import ExperimentRunner
from .results import StudyResult, TrialResult
from .space import generate_trial_config, search_space

Objective = Callable[[dict[str, Any]], float]


def _objective_from_runner(config: dict[str, Any]) -> float:
    result = ExperimentRunner().run(experiment_config_from_mapping(config))
    if result.status.value != "succeeded":
        raise RuntimeError(result.error or "canonical trial failed")
    if result.training.best_metric is None:
        raise RuntimeError(
            "Canonical tuning requires a validation best metric; test evaluation "
            "is never used as a tuning objective."
        )
    return float(result.training.best_metric.value)


def run_study(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    output_dir: str | Path = "results/optuna",
    study_name: str = "canonical-study",
    n_trials: int | None = None,
    objective: Objective | None = None,
) -> StudyResult:
    """Run a study through canonical config and runner factories.

    Optuna is loaded only when a real study is requested.  If the optional
    dependency is absent, a deterministic local sampler still provides a
    useful CPU-compatible synthetic study and structured result file.
    """

    base = config.to_mapping() if isinstance(config, ExperimentConfig) else dict(config)
    tuning = base.get("tuning", {})
    count = int(n_trials if n_trials is not None else tuning.get("n_trials", 1))
    if count <= 0:
        raise ValueError("n_trials must be positive")
    parameters = search_space(base)
    direction = str(tuning.get("direction", "maximize")).lower()
    metric = str(tuning.get("metric", "val_accuracy"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = StudyResult(
        study_name=study_name,
        algorithm=str(base.get("algorithm", {}).get("name", "bp")).lower(),
        direction=direction,
        metric=metric,
    )
    evaluate = objective or _objective_from_runner

    try:
        import optuna
    except ImportError:
        optuna = None

    if optuna is not None:
        storage = f"sqlite:///{(output / f'{study_name}.db').resolve()}"
        sampler_name = str(tuning.get("sampler", "TPE")).upper()
        sampler = (
            optuna.samplers.RandomSampler(
                seed=int(base.get("general", {}).get("seed", 42))
            )
            if sampler_name == "RANDOM"
            else optuna.samplers.TPESampler(
                seed=int(base.get("general", {}).get("seed", 42))
            )
        )
        pruner_name = str(tuning.get("pruner", "Median")).upper()
        pruner = (
            optuna.pruners.NopPruner()
            if pruner_name == "NONE"
            else optuna.pruners.MedianPruner()
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        def optuna_objective(trial: Any) -> float:
            params = {
                parameter.name: parameter.suggest(trial) for parameter in parameters
            }
            trial_config = generate_trial_config(base, params, trial.number)
            value = float(evaluate(trial_config))
            trial.set_user_attr("trial_seed", trial_config["general"]["seed"])
            return value

        study.optimize(optuna_objective, n_trials=count)
        for trial in study.trials:
            if trial.value is None:
                continue
            result.trials.append(
                TrialResult(
                    number=trial.number,
                    seed=int(
                        trial.user_attrs.get(
                            "trial_seed",
                            base.get("general", {}).get("seed", 42) + trial.number,
                        )
                    ),
                    value=float(trial.value),
                    params=dict(trial.params),
                    state=str(trial.state).lower(),
                )
            )
        if study.best_trial is not None:
            result.best_trial_number = study.best_trial.number
            result.best_value = float(study.best_value)
            result.best_params = dict(study.best_trial.params)
    else:
        random = Random(int(base.get("general", {}).get("seed", 42)))
        for number in range(count):
            params = {
                parameter.name: parameter.sample(random) for parameter in parameters
            }
            trial_config = generate_trial_config(base, params, number)
            try:
                value = float(evaluate(trial_config))
                trial = TrialResult(
                    number, trial_config["general"]["seed"], value, params
                )
            except Exception as exc:
                trial = TrialResult(
                    number,
                    trial_config["general"]["seed"],
                    None,
                    params,
                    "failed",
                    str(exc),
                )
            result.trials.append(trial)
        complete = [trial for trial in result.trials if trial.value is not None]
        if complete:

            def trial_value(item: TrialResult) -> float:
                assert item.value is not None
                return item.value

            best = (
                max(complete, key=trial_value)
                if direction == "maximize"
                else min(complete, key=trial_value)
            )
            result.best_trial_number, result.best_value, result.best_params = (
                best.number,
                best.value,
                best.params,
            )

    (output / f"{study_name}.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


__all__ = ["run_study"]
