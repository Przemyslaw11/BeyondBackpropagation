"""Optuna objective for the joint-gradient rungs of the ablation ladder.

BP-DS and MF-Joint search the same breadth as BP and MF over learning rate and
weight decay, plus ``aux_weight``. That third dimension exists only because
summing the per-layer losses into one objective forces a choice of weighting that
layer-sequential MF never has to make.
"""

import copy
import logging
import pprint
import time
from typing import Any, Callable, Dict, Tuple

import optuna
import torch

from src.baselines.bp_ds import (
    evaluate_bp_ds_model,
    train_bp_ds_model,
)
from src.data_utils.datasets import get_dataloaders
from src.training.engine import get_model_and_adapter
from src.utils.backend_policy import get_execution_backend
from src.utils.early_stopping import resolve_tuning_max_epochs
from src.utils.helpers import format_time, set_seed

logger = logging.getLogger(__name__)

DEFAULT_AUX_WEIGHT_RANGE = (0.01, 10.0)


def _setup_trial(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], torch.device, int]:
    """Suggests hyperparameters and prepares the environment for one trial."""
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration.")

    cfg.setdefault("optimizer", {})
    cfg.setdefault("algorithm_params", {})

    cfg["optimizer"]["lr"] = trial.suggest_float(
        "lr", *tuning_cfg.get("lr_range", [1e-5, 1e-2]), log=True
    )
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "wd", *tuning_cfg.get("wd_range", [1e-6, 1e-3]), log=True
    )
    cfg["algorithm_params"]["aux_weight"] = trial.suggest_float(
        "aux_weight",
        *tuning_cfg.get("aux_weight_range", DEFAULT_AUX_WEIGHT_RANGE),
        log=True,
    )
    cfg.setdefault("early_stopping", {})["max_epochs"] = resolve_tuning_max_epochs(cfg)

    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    backend = get_execution_backend(cfg)
    device = backend.resolve_device(cfg.get("general", {}).get("device", "auto"))
    return cfg, device, trial_seed


def objective_deep_supervised(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    train_fn: Callable,
    evaluate_fn: Callable,
    label: str,
) -> float:
    """Trains one rung to completion and returns its validation accuracy."""
    cfg, device, trial_seed = _setup_trial(trial, base_config)
    tuning_cfg = cfg["tuning"]
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()

    logger.info(
        f"--- Starting Optuna Trial {trial.number} "
        f"(Study: {trial.study.study_name}) for {label} ---"
    )
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    logger.info(f"  {label} Hyperparameters:\n{pprint.pformat(trial.params)}")

    model = None
    try:
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=trial_seed,
            config=cfg,
            backend=cfg.get("general", {}).get("backend", "slurm"),
            download=data_config.get("download", True),
        )
        if not val_loader:
            raise ValueError(f"{label} tuning needs a validation split.")

        model, input_adapter = get_model_and_adapter(cfg, device)
        model.to(device)

        trial_train_start_time = time.time()
        train_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            device=device,
            wandb_run=None,
            input_adapter=input_adapter,
            step_ref=[-1],
            gpu_handle=None,
            nvml_active=False,
        )
        logger.info(
            f"{label} training completed in "
            f"{format_time(time.time() - trial_train_start_time)}."
        )

        _, validation_accuracy = evaluate_fn(
            model, val_loader, torch.nn.CrossEntropyLoss(), device, input_adapter
        )
        if torch.isnan(torch.tensor(validation_accuracy)):
            raise ValueError("Evaluation returned NaN accuracy.")

        logger.info(
            f"Trial {trial.number} finished. "
            f"Final Validation Accuracy: {validation_accuracy:.4f}"
        )
        return validation_accuracy

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        return -1.0 if optimization_direction == "maximize" else float("inf")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"--- Finished Optuna Trial {trial.number} for {label} ---")


def objective_bpds(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Rung 2: backpropagation with matching layer-wise auxiliary losses."""
    return objective_deep_supervised(
        trial, base_config, train_bp_ds_model, evaluate_bp_ds_model, "BP-DS"
    )
