import copy
import logging
import pprint
import time
from typing import Any, Dict

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

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


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter tuning of BP baselines.
    MODIFIED: Added logging of trial info to main logger.
    """
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

    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) ---")
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    param_str = pprint.pformat(trial.params)
    logger.info(f"  Hyperparameters:\n{param_str}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize not in ["val_accuracy", "val_loss"]:
        logger.warning(
            f"Unsupported optimization metric '{metric_to_optimize}'. Defaulting to 'val_accuracy'."
        )
        metric_to_optimize = "val_accuracy"

    model = None
    optimizer = None
    train_loader = None
    val_loader = None
    criterion = None

    try:
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info(f"Trial {trial.number}: Loading data...")
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
            raise ValueError(
                "Validation loader is required for Optuna tuning but was not created."
            )
        logger.info(f"Trial {trial.number}: Data loaded.")

        logger.info(f"Trial {trial.number}: Instantiating BP baseline model and getting adapter...")
        if cfg.get("algorithm", {}).get("name", "").lower() != "bp":
            logger.warning(f"Trial {trial.number}: Overriding algorithm to 'BP' for baseline tuning.")
            cfg["algorithm"] = {"name": "BP"}

        model, input_adapter = get_model_and_adapter(
            cfg, device
        )
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' baseline ({num_params:,} params) on {device}. "
            f"Input adapter type: {type(input_adapter)}"
        )

        optimizer_cfg = cfg.get("optimizer", {})
        optimizer_type = optimizer_cfg.get("type", "AdamW").lower()

        opt_kwargs = {
            "lr": trial.params["lr"],
            "weight_decay": trial.params["wd"],
        }
        if optimizer_type == "sgd":
            opt_kwargs["momentum"] = trial.params.get(
                "momentum", 0.9
            )

        if optimizer_type == "adamw":
            optimizer = optim.AdamW(model.parameters(), **opt_kwargs)
        elif optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), **opt_kwargs)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), **opt_kwargs)
        else:
            raise ValueError(
                f"Unsupported optimizer type specified in config: {optimizer_type}"
            )

        criterion_name = cfg.get("training", {}).get("criterion", "CrossEntropyLoss")
        if criterion_name.lower() == "crossentropyloss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unsupported criterion specified in config: {criterion_name}"
            )
        logger.info(
            f"Trial {trial.number}: Optimizer ({optimizer_type}) and Criterion ({criterion_name}) setup."
        )

        num_epochs = tuning_cfg.get("num_epochs", 10)
        best_val_metric_for_trial = (
            -float("inf") if optimization_direction == "maximize" else float("inf")
        )

        logger.info(f"Trial {trial.number}: Starting training loop for {num_epochs} epochs.")
        trial_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss, train_acc, _ = train_bp_epoch(
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
                f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Metric ({metric_to_optimize}): {current_val_metric:.4f} | "
                f"Duration: {format_time(epoch_duration)}"
            )

            trial.report(current_val_metric, epoch)

            if trial.should_prune():
                logger.warning(f"Trial {trial.number} pruned at epoch {epoch+1}.")
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
            f"Trial {trial.number} finished. Duration: {format_time(total_trial_time)}. "
            f"Best Value ({metric_to_optimize}): {best_val_metric_for_trial:.4f}"
        )

        return best_val_metric_for_trial

    except optuna.TrialPruned as e:
        raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        return -float("inf") if optimization_direction == "maximize" else float("inf")
    finally:
        del model, optimizer, train_loader, val_loader, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")