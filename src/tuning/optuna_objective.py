# File: src/tuning/optuna_objective.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from typing import Dict, Any, Optional

# --- Import necessary components from your project ---
from src.utils.helpers import set_seed, format_time

# Assuming config_parser handles merging correctly
# from src.utils.config_parser import load_config
from src.data_utils.datasets import get_dataloaders

# Import model creation logic (same as used in engine.py)
from src.training.engine import get_model_and_adapter

# Import the actual BP training/evaluation functions
from src.baselines.bp import train_bp_epoch, evaluate_bp_model

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, base_config: dict) -> float:
    """
    Optuna objective function for hyperparameter tuning of BP baselines.
    Trains a model instance with suggested hyperparameters and returns
    the validation accuracy.

    Args:
        trial: An Optuna Trial object providing suggested hyperparameters.
        base_config: The base configuration dictionary loaded from YAML,
                     potentially already merged with experiment specifics.

    Returns:
        The metric to optimize (e.g., best validation accuracy achieved).
    """
    # --- Hyperparameter Suggestion & Config Setup ---
    # Create a mutable copy of the config for this trial
    cfg = {}
    # Deep copy necessary parts to avoid modifying the original dict across trials
    for key, value in base_config.items():
        if isinstance(value, dict):
            cfg[key] = value.copy()  # Shallow copy top-level dicts
        else:
            cfg[key] = value
    # Further deep copies might be needed if nested structures are modified

    # Suggest hyperparameters using Optuna trial object
    # Make sure 'tuning' section exists in the config
    tuning_cfg = cfg.get("tuning", {})
    if not tuning_cfg:
        logger.error(
            "Optuna objective requires a 'tuning' section in the configuration."
        )
        raise ValueError("Missing 'tuning' section in config for Optuna.")

    # Ensure 'optimizer' section exists for modification
    if "optimizer" not in cfg:
        cfg["optimizer"] = {}

    cfg["optimizer"]["lr"] = trial.suggest_float(
        "lr",
        tuning_cfg.get("lr_range", [1e-5, 1e-2])[0],  # Default range
        tuning_cfg.get("lr_range", [1e-5, 1e-2])[1],
        log=True,
    )
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        tuning_cfg.get("wd_range", [1e-6, 1e-1])[0],  # Default range
        tuning_cfg.get("wd_range", [1e-6, 1e-1])[1],
        log=True,
    )
    # Example: Optionally tune batch size
    # if 'batch_size_range' in tuning_cfg:
    #     if 'data_loader' not in cfg: cfg['data_loader'] = {}
    #     cfg['data_loader']['batch_size'] = trial.suggest_categorical(
    #         'batch_size', tuning_cfg['batch_size_range']
    #     )

    # --- Setup (Seed, Device) ---
    trial_seed = base_config.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = base_config.get("general", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)

    logger.info(f"--- Starting Optuna Trial {trial.number} ---")
    logger.info(f"Device: {device}, Seed: {trial_seed}")
    logger.info(
        f"Hyperparameters: lr={cfg['optimizer']['lr']:.6f}, wd={cfg['optimizer']['weight_decay']:.6f}"
    )
    # logger.info(f"Batch Size: {cfg.get('data_loader', {}).get('batch_size', 'N/A')}") # If tuning batch size

    try:
        # --- Data Loading ---
        data_config = cfg.get("dataset", {})
        loader_config = cfg.get("data_loader", {})
        logger.info("Loading data...")
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root_dir", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=trial_seed,  # Use trial-specific seed for split consistency if needed
            num_workers=loader_config.get(
                "num_workers", 0
            ),  # Use 0 for Optuna trials? Avoids multiprocessing issues.
            pin_memory=(
                loader_config.get("pin_memory", False) if device.type == "cpu" else True
            ),
            download=data_config.get("download", True),
        )
        if not val_loader:
            logger.error(
                f"Trial {trial.number}: Validation loader is missing. Cannot perform tuning."
            )
            raise ValueError("Validation set required for Optuna tuning.")
        logger.info("Data loaded.")

        # --- Model Instantiation ---
        # Use the same logic as the main training engine to get the correct BP baseline model
        logger.info("Instantiating model...")
        # Pass the trial-modified config `cfg` here
        model, input_adapter = get_model_and_adapter(
            cfg
        )  # Assumes get_model_and_adapter uses cfg['model']
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model '{cfg.get('model', {}).get('name')}' ({num_params:,} params) instantiated on {device}."
        )

        # --- Optimizer and Criterion ---
        optimizer_cfg = cfg.get("optimizer", {})
        optimizer_type = optimizer_cfg.get("type", "AdamW").lower()

        # Instantiate optimizer with hyperparameters suggested by Optuna
        if optimizer_type == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=optimizer_cfg["lr"],
                weight_decay=optimizer_cfg["weight_decay"],
            )
        elif optimizer_type == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_cfg["lr"],
                weight_decay=optimizer_cfg["weight_decay"],
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=optimizer_cfg["lr"],
                momentum=optimizer_cfg.get("momentum", 0.9),
                weight_decay=optimizer_cfg["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer type for tuning: {optimizer_type}")

        criterion_name = cfg.get("training", {}).get("criterion", "CrossEntropyLoss")
        if criterion_name.lower() == "crossentropyloss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion for tuning: {criterion_name}")
        logger.info(
            f"Optimizer ({optimizer_type}) and Criterion ({criterion_name}) setup."
        )

        # --- Training & Evaluation Loop for the Trial ---
        num_epochs = tuning_cfg.get(
            "num_epochs", 10
        )  # Number of epochs for this Optuna trial
        best_val_metric = -1.0  # Initialize for maximization (e.g., accuracy)
        metric_to_optimize = tuning_cfg.get(
            "metric", "val_accuracy"
        ).lower()  # 'val_accuracy' or 'val_loss'
        optimization_direction = tuning_cfg.get("direction", "maximize").lower()
        if optimization_direction == "minimize":
            best_val_metric = float("inf")

        logger.info(f"Starting training loop for {num_epochs} epochs.")
        trial_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # --- Call actual training function for one epoch ---
            train_loss, train_acc = train_bp_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,  # Pass epoch for potential internal logging if needed
                wandb_run=None,  # Disable wandb logging for individual trials (too verbose)
                log_interval=99999,  # Suppress batch logging within epoch train
                input_adapter=input_adapter,
            )

            # --- Call actual evaluation function on validation set ---
            val_loss, val_acc = evaluate_bp_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                input_adapter=input_adapter,
            )

            epoch_duration = time.time() - epoch_start_time

            # --- Select the metric Optuna is optimizing ---
            if metric_to_optimize == "val_accuracy":
                current_val_metric = val_acc
            elif metric_to_optimize == "val_loss":
                current_val_metric = val_loss
            else:
                logger.error(
                    f"Unsupported metric to optimize: {metric_to_optimize}. Defaulting to val_accuracy."
                )
                current_val_metric = val_acc

            logger.info(
                f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Metric ({metric_to_optimize}): {current_val_metric:.4f} | "
                f"Duration: {format_time(epoch_duration)}"
            )

            # --- Optuna Pruning & Reporting ---
            trial.report(current_val_metric, epoch)
            if trial.should_prune():
                logger.warning(
                    f"Trial {trial.number} pruned at epoch {epoch+1} based on {metric_to_optimize}."
                )
                raise optuna.TrialPruned()

            # --- Update best metric for this trial ---
            if optimization_direction == "maximize":
                best_val_metric = max(best_val_metric, current_val_metric)
            else:  # minimize
                best_val_metric = min(best_val_metric, current_val_metric)

        # --- End of Trial ---
        total_trial_time = time.time() - trial_start_time
        logger.info(
            f"Trial {trial.number} finished. Duration: {format_time(total_trial_time)}. Best Value ({metric_to_optimize}): {best_val_metric:.4f}"
        )

        # Return the final metric Optuna should optimize (best value achieved across epochs)
        return best_val_metric

    except optuna.TrialPruned:
        # Re-raise prune exceptions for Optuna to handle
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        # Return a value indicating failure based on optimization direction
        return 0.0 if optimization_direction == "maximize" else float("inf")
    finally:
        # Clean up GPU memory if needed (optional)
        del model, optimizer, train_loader, val_loader  # Release references
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")
