# File: src/tuning/optuna_objective.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import copy
from typing import Dict, Any, Optional, Tuple, Callable

from src.utils.helpers import set_seed, format_time
from src.data_utils.datasets import get_dataloaders
from src.training.engine import get_model_and_adapter
from src.baselines.bp import train_bp_epoch, evaluate_bp_model

logger = logging.getLogger(__name__)  # Get logger instance


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter tuning of BP baselines.
    """
    # --- Hyperparameter Suggestion & Config Setup ---
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section.")
    if "optimizer" not in cfg or not isinstance(cfg["optimizer"], dict):
        cfg["optimizer"] = {}

    # Suggest hyperparameters based on config ranges
    cfg["optimizer"]["lr"] = trial.suggest_float(
        "lr", *tuning_cfg.get("lr_range", [1e-5, 1e-2]), log=True
    )
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "wd", *tuning_cfg.get("wd_range", [1e-6, 1e-3]), log=True
    )  # Renamed trial param to 'wd'
    optimizer_type = cfg.get("optimizer", {}).get("type", "AdamW").lower()
    if optimizer_type == "sgd":
        cfg["optimizer"]["momentum"] = trial.suggest_float(
            "momentum", *tuning_cfg.get("momentum_range", [0.8, 0.99])
        )

    # --- Setup (Seed, Device) ---
    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"--- Starting Optuna Trial {trial.number} ---")
    logger.info(f"Device: {device}, Seed: {trial_seed}")
    logger.info(f"Hyperparameters: {trial.params}")  # Log suggested params

    try:
        # --- Data Loading ---
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info("Loading data for Optuna trial...")
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=trial_seed,
            num_workers=0,  # Use 0 workers for Optuna trials
            pin_memory=False,  # Disable pin_memory
            download=data_config.get("download", True),
        )
        if not val_loader:
            raise ValueError("Validation loader missing.")
        logger.info("Data loaded.")

        # --- Model Instantiation ---
        logger.info("Instantiating BP baseline model...")
        cfg["algorithm"] = {"name": "BP"}  # Ensure BP baseline creation
        model, input_adapter = get_model_and_adapter(cfg, device)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model '{cfg.get('model', {}).get('name')}' baseline ({num_params:,} params) on {device}."
        )

        # --- Optimizer and Criterion ---
        optimizer_cfg = cfg.get("optimizer", {})
        optimizer_type = optimizer_cfg.get("type", "AdamW").lower()

        # Use suggested parameters
        opt_kwargs = {
            "lr": trial.params["lr"],
            "weight_decay": trial.params["wd"],  # Use trial param name 'wd'
        }
        if optimizer_type == "sgd":
            opt_kwargs["momentum"] = trial.params.get(
                "momentum", 0.9
            )  # Get momentum if suggested

        if optimizer_type == "adamw":
            optimizer = optim.AdamW(model.parameters(), **opt_kwargs)
        elif optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), **opt_kwargs)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), **opt_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        criterion_name = cfg.get("training", {}).get("criterion", "CrossEntropyLoss")
        if criterion_name.lower() == "crossentropyloss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
        logger.info(
            f"Optimizer ({optimizer_type}) and Criterion ({criterion_name}) setup."
        )

        # --- Training & Evaluation Loop ---
        num_epochs = tuning_cfg.get("num_epochs", 10)
        best_val_metric_for_trial = (
            -float("inf")
            if tuning_cfg.get("direction", "maximize").lower() == "maximize"
            else float("inf")
        )
        metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
        optimization_direction = tuning_cfg.get("direction", "maximize").lower()

        logger.info(f"Starting trial training loop for {num_epochs} epochs.")
        trial_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = train_bp_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                num_epochs,
                None,
                99999,
                input_adapter,
            )
            val_loss, val_acc = evaluate_bp_model(
                model, val_loader, criterion, device, input_adapter
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
                # Return intermediate value upon pruning
                return current_val_metric  # Optuna uses this for pruning decision effectiveness

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
            f"Trial {trial.number} finished. Duration: {format_time(total_trial_time)}. Best Value ({metric_to_optimize}): {best_val_metric_for_trial:.4f}"
        )
        return best_val_metric_for_trial

    except optuna.TrialPruned as e:
        logger.info(f"Trial {trial.number} pruned.")
        raise e  # Re-raise prune exceptions
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return (
            -float("inf") if optimization_direction == "maximize" else float("inf")
        )  # Return poor value
    finally:
        del model, optimizer, train_loader, val_loader, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")
