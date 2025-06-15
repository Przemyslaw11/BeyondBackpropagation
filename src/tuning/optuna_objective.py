"""Optuna objective function for standard backpropagation (BP) model tuning."""

import copy
import logging
import pprint
import time
from typing import Any, Callable, Dict, Optional

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.baselines.bp import (
    evaluate_bp_model,
    train_bp_epoch,
)
from src.data_utils.datasets import get_dataloaders
from src.training.engine import (
    get_model_and_adapter,
)
from src.utils.helpers import format_time, set_seed

logger = logging.getLogger(__name__)


def _get_trial_config(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Suggests hyperparameters and updates the configuration for a trial."""
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration.")
    if "optimizer" not in cfg or not isinstance(cfg["optimizer"], dict):
        cfg["optimizer"] = {}

    cfg["optimizer"]["lr"] = trial.suggest_float(
        "lr", *tuning_cfg.get("lr_range", [1e-5, 1e-2]), log=True
    )
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "wd", *tuning_cfg.get("wd_range", [1e-6, 1e-3]), log=True
    )
    optimizer_type = cfg.get("optimizer", {}).get("type", "AdamW").lower()
    if optimizer_type == "sgd":
        cfg["optimizer"]["momentum"] = trial.suggest_float(
            "momentum", *tuning_cfg.get("momentum_range", [0.8, 0.99])
        )
    return cfg


def _get_device(cfg: Dict[str, Any]) -> torch.device:
    """Sets up the device for a trial."""
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_bp_optimizer(
    model: nn.Module, trial_params: Dict[str, Any], optimizer_cfg: Dict[str, Any]
) -> optim.Optimizer:
    """Creates an optimizer for the model based on trial parameters."""
    optimizer_type = optimizer_cfg.get("type", "AdamW").lower()
    opt_kwargs = {"lr": trial_params["lr"], "weight_decay": trial_params["wd"]}
    if optimizer_type == "sgd":
        opt_kwargs["momentum"] = trial_params.get("momentum", 0.9)

    if optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), **opt_kwargs)
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), **opt_kwargs)
    if optimizer_type == "sgd":
        return optim.SGD(model.parameters(), **opt_kwargs)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def _run_bp_trial_epoch_loop(
    trial: optuna.Trial,
    cfg: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> float:
    """Runs the training and evaluation loop for one BP trial."""
    tuning_cfg = cfg["tuning"]
    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    num_epochs = tuning_cfg.get("num_epochs", 10)
    best_val_metric_for_trial = (
        -float("inf") if optimization_direction == "maximize" else float("inf")
    )

    logger.info(
        f"Trial {trial.number}: Starting training loop for {num_epochs} epochs."
    )
    trial_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_bp_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=num_epochs,
            wandb_run=None,
            log_interval=99999,
            input_adapter=input_adapter,
            step_ref=[-1],
            gpu_handle=None,
            nvml_active=False,
        )
        val_loss, val_acc = evaluate_bp_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            input_adapter=input_adapter,
        )
        epoch_duration = time.time() - epoch_start_time
        current_val_metric = (
            val_acc if metric_to_optimize == "val_accuracy" else val_loss
        )

        logger.info(
            f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"Metric ({metric_to_optimize}): {current_val_metric:.4f} | "
            f"Duration: {format_time(epoch_duration)}"
        )

        trial.report(current_val_metric, epoch)
        if trial.should_prune():
            logger.warning(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
            raise optuna.TrialPruned()

        if optimization_direction == "maximize":
            best_val_metric_for_trial = max(
                best_val_metric_for_trial, current_val_metric
            )
        else:
            best_val_metric_for_trial = min(
                best_val_metric_for_trial, current_val_metric
            )

    total_trial_time = time.time() - trial_start_time
    logger.info(
        f"Trial {trial.number} finished. "
        f"Duration: {format_time(total_trial_time)}. "
        f"Best Value ({metric_to_optimize}): {best_val_metric_for_trial:.4f}"
    )
    return best_val_metric_for_trial


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter tuning of BP baselines."""
    cfg = _get_trial_config(trial, base_config)
    tuning_cfg = cfg["tuning"]

    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device = _get_device(cfg)

    logger.info(
        f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) ---"
    )
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    logger.info(f"  Hyperparameters:\n{pprint.pformat(trial.params)}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    if metric_to_optimize not in ["val_accuracy", "val_loss"]:
        logger.warning(
            f"Unsupported metric '{metric_to_optimize}'. Defaulting to 'val_accuracy'."
        )
        metric_to_optimize = "val_accuracy"
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()

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
            num_workers=0,
            pin_memory=False,
            download=data_config.get("download", True),
        )
        if not val_loader:
            raise ValueError("Validation loader is required for Optuna tuning.")
        logger.info(f"Trial {trial.number}: Data loaded.")

        if cfg.get("algorithm", {}).get("name", "").lower() != "bp":
            logger.warning("Overriding algorithm to 'BP' for baseline tuning.")
            cfg["algorithm"] = {"name": "BP"}
        model, input_adapter = get_model_and_adapter(cfg, device)
        model.to(device)

        optimizer = _create_bp_optimizer(model, trial.params, cfg.get("optimizer", {}))
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Trial {trial.number}: Optimizer and Criterion setup.")

        return _run_bp_trial_epoch_loop(
            trial,
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            input_adapter,
        )

    except optuna.TrialPruned as e:
        raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        return -float("inf") if optimization_direction == "maximize" else float("inf")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")
