# File: src/tuning/optuna_objective.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import copy  # For deep copying config
from typing import Dict, Any, Optional, Tuple, Callable  # Added Callable

from src.utils.helpers import set_seed, format_time
from src.data_utils.datasets import get_dataloaders
from src.training.engine import get_model_and_adapter  # Re-use model creation
from src.baselines.bp import train_bp_epoch, evaluate_bp_model  # Use BP functions

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter tuning of BP baselines.
    Trains a model instance with suggested hyperparameters and returns
    the validation metric.

    Args:
        trial: An Optuna Trial object providing suggested hyperparameters.
        base_config: The base configuration dictionary loaded from YAML,
                     potentially already merged with experiment specifics.

    Returns:
        The metric to optimize (e.g., best validation accuracy).
    """
    # --- Hyperparameter Suggestion & Config Setup ---
    # Create a deep copy to avoid modifying the original across trials
    cfg = copy.deepcopy(base_config)

    # Ensure 'tuning' section exists
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        logger.error("Optuna objective requires a 'tuning' dictionary in the config.")
        raise ValueError("Missing or invalid 'tuning' section in config.")

    # Ensure 'optimizer' section exists for modification (create if needed)
    if "optimizer" not in cfg or not isinstance(cfg["optimizer"], dict):
        cfg["optimizer"] = {}

    # Suggest hyperparameters using Optuna trial object
    cfg["optimizer"]["lr"] = trial.suggest_float(
        "lr",
        tuning_cfg.get("lr_range", [1e-5, 1e-2])[0],
        tuning_cfg.get("lr_range", [1e-5, 1e-2])[1],
        log=True,
    )
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        tuning_cfg.get("wd_range", [1e-6, 1e-3])[0],  # Adjusted default upper WD limit
        tuning_cfg.get("wd_range", [1e-6, 1e-3])[1],
        log=True,
    )
    # Add suggestions for other parameters if needed (e.g., momentum for SGD)
    # if cfg["optimizer"].get("type", "AdamW").lower() == "sgd":
    #     cfg["optimizer"]["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)

    # --- Setup (Seed, Device) ---
    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)

    logger.info(f"--- Starting Optuna Trial {trial.number} ---")
    logger.info(f"Device: {device}, Seed: {trial_seed}")
    logger.info(
        f"Hyperparameters: lr={cfg['optimizer']['lr']:.6f}, wd={cfg['optimizer']['weight_decay']:.6f}"
    )

    try:
        # --- Data Loading ---
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info("Loading data for Optuna trial...")
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get(
                "batch_size", 64
            ),  # Use batch size from config
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=trial_seed,
            num_workers=loader_config.get(
                "num_workers", 0
            ),  # Use 0 for Optuna trials to avoid issues
            pin_memory=False,  # Disable pin_memory for Optuna trials simplicity
            download=data_config.get("download", True),
        )
        if not val_loader:
            logger.error(
                f"Trial {trial.number}: Validation loader missing. Cannot tune."
            )
            raise ValueError("Validation set required for Optuna tuning.")
        logger.info("Data loaded.")

        # --- Model Instantiation ---
        logger.info("Instantiating BP baseline model...")
        # Ensure algorithm is set to 'BP' for correct baseline creation by get_model_and_adapter
        cfg["algorithm"] = {"name": "BP"}
        model, input_adapter = get_model_and_adapter(
            cfg, device
        )  # Pass trial config and device
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model '{cfg.get('model', {}).get('name')}' baseline ({num_params:,} params) on {device}."
        )

        # --- Optimizer and Criterion ---
        optimizer_cfg = cfg.get("optimizer", {})
        optimizer_type = optimizer_cfg.get("type", "AdamW").lower()

        opt_params = {
            "lr": optimizer_cfg["lr"],
            "weight_decay": optimizer_cfg["weight_decay"],
        }
        if optimizer_type == "sgd":
            opt_params["momentum"] = optimizer_cfg.get("momentum", 0.9)

        if optimizer_type == "adamw":
            optimizer = optim.AdamW(model.parameters(), **opt_params)
        elif optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), **opt_params)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), **opt_params)
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

        # --- Training & Evaluation Loop for the Trial ---
        num_epochs = tuning_cfg.get("num_epochs", 10)  # Epochs per Optuna trial
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

            # --- Training Epoch ---
            train_loss, train_acc = train_bp_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=num_epochs,
                wandb_run=None,  # No W&B per trial
                log_interval=99999,
                input_adapter=input_adapter,
            )

            # --- Validation Epoch ---
            val_loss, val_acc = evaluate_bp_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                input_adapter=input_adapter,
            )

            epoch_duration = time.time() - epoch_start_time

            # --- Select metric for Optuna ---
            current_val_metric = (
                val_acc if metric_to_optimize == "val_accuracy" else val_loss
            )

            logger.info(
                f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Metric ({metric_to_optimize}): {current_val_metric:.4f} | "
                f"Duration: {format_time(epoch_duration)}"
            )

            # --- Optuna Pruning & Reporting ---
            trial.report(current_val_metric, epoch)
            if trial.should_prune():
                logger.warning(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                raise optuna.TrialPruned()

            # --- Update best metric *for this trial* ---
            if optimization_direction == "maximize":
                best_val_metric_for_trial = max(
                    best_val_metric_for_trial, current_val_metric
                )
            else:
                best_val_metric_for_trial = min(
                    best_val_metric_for_trial, current_val_metric
                )

        # --- End of Trial ---
        total_trial_time = time.time() - trial_start_time
        logger.info(
            f"Trial {trial.number} finished. Duration: {format_time(total_trial_time)}. Best Value ({metric_to_optimize}): {best_val_metric_for_trial:.4f}"
        )

        # Return the best metric value achieved during this trial
        return best_val_metric_for_trial

    except optuna.TrialPruned:
        raise  # Re-raise prune exceptions
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        # Return a poor value to Optuna indicating failure
        return -float("inf") if optimization_direction == "maximize" else float("inf")
    finally:
        # Clean up GPU memory (important for Optuna loops)
        del model, optimizer, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
            # logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")
