# File: src/tuning/optuna_objective.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import copy
import pprint # For pretty printing dicts
from typing import Dict, Any, Optional, Tuple, Callable

from src.utils.helpers import set_seed, format_time
from src.data_utils.datasets import get_dataloaders
from src.training.engine import (
    get_model_and_adapter,
)  # Import the function that provides the adapter
from src.baselines.bp import (
    train_bp_epoch,
    evaluate_bp_model,
)  # Keep evaluate_bp_model signature consistent

# Use the centrally configured logger
logger = logging.getLogger(__name__) # Get logger instance


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter tuning of BP baselines.
    MODIFIED: Added logging of trial info to main logger.
    """
    # --- Hyperparameter Suggestion & Config Setup ---
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration.")
    # Ensure optimizer sub-dict exists
    if "optimizer" not in cfg or not isinstance(cfg["optimizer"], dict):
        cfg["optimizer"] = {}

    # Suggest hyperparameters based on config ranges defined in the 'tuning' section
    cfg["optimizer"]["lr"] = trial.suggest_float(
        "lr", *tuning_cfg.get("lr_range", [1e-5, 1e-2]), log=True
    )
    cfg["optimizer"]["weight_decay"] = trial.suggest_float(
        "wd", *tuning_cfg.get("wd_range", [1e-6, 1e-3]), log=True
    )  # Use 'wd' for trial name
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

    # <<< ADDED: Log trial start info to main logger >>>
    logger.info(f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) ---")
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    param_str = pprint.pformat(trial.params)
    logger.info(f"  Hyperparameters:\n{param_str}")
    # --- END ADDED Logging ---

    # Define optimization direction and metric from config
    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize not in ["val_accuracy", "val_loss"]:
        logger.warning(
            f"Unsupported optimization metric '{metric_to_optimize}'. Defaulting to 'val_accuracy'."
        )
        metric_to_optimize = "val_accuracy"

    model = None  # Initialize for finally block
    optimizer = None
    train_loader = None
    val_loader = None
    criterion = None

    try:
        # --- Data Loading ---
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info(f"Trial {trial.number}: Loading data...")
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=trial_seed,
            # Use 0 workers and disable pin_memory for simpler Optuna runs
            num_workers=0,
            pin_memory=False,
            download=data_config.get("download", True),
        )
        if not val_loader:
            raise ValueError(
                "Validation loader is required for Optuna tuning but was not created."
            )
        logger.info(f"Trial {trial.number}: Data loaded.")

        # --- Model Instantiation & Input Adapter Retrieval ---
        logger.info(f"Trial {trial.number}: Instantiating BP baseline model and getting adapter...")
        # Ensure we are creating the BP baseline version of the model
        # The logic inside get_model_and_adapter handles this if algorithm is 'BP'
        if cfg.get("algorithm", {}).get("name", "").lower() != "bp":
            logger.warning(f"Trial {trial.number}: Overriding algorithm to 'BP' for baseline tuning.")
            cfg["algorithm"] = {"name": "BP"}

        model, input_adapter = get_model_and_adapter(
            cfg, device
        )  # Get model AND adapter here
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' baseline ({num_params:,} params) on {device}. "
            f"Input adapter type: {type(input_adapter)}"
        )

        # --- Optimizer and Criterion ---
        optimizer_cfg = cfg.get("optimizer", {})
        optimizer_type = optimizer_cfg.get("type", "AdamW").lower()

        # Use suggested parameters from the trial
        opt_kwargs = {
            "lr": trial.params["lr"],
            "weight_decay": trial.params["wd"],  # Use trial param name 'wd'
        }
        if optimizer_type == "sgd":
            opt_kwargs["momentum"] = trial.params.get(
                "momentum", 0.9
            )  # Use trial momentum if available

        # Select optimizer based on type
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

        # --- Training & Evaluation Loop ---
        num_epochs = tuning_cfg.get("num_epochs", 10)  # Epochs per trial
        best_val_metric_for_trial = (
            -float("inf") if optimization_direction == "maximize" else float("inf")
        )

        logger.info(f"Trial {trial.number}: Starting training loop for {num_epochs} epochs.")
        trial_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            # Pass the retrieved input_adapter to the training epoch function
            train_loss, train_acc, _ = train_bp_epoch( # Ignore peak mem return here
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=num_epochs,
                wandb_run=None,  # No W&B logging for Optuna trials by default
                log_interval=99999,  # Minimize logging noise during tuning
                input_adapter=input_adapter,  # Pass the adapter
                step_ref=[-1], # Step ref not needed here
                gpu_handle=None, # No monitoring during Optuna
                nvml_active=False,
            )
            # Pass the retrieved input_adapter to the evaluation function
            val_loss, val_acc = evaluate_bp_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                input_adapter=input_adapter,  # Pass the adapter
            )
            epoch_duration = time.time() - epoch_start_time

            # Determine the metric value for this epoch based on config
            current_val_metric = (
                val_acc if metric_to_optimize == "val_accuracy" else val_loss
            )

            logger.info(
                f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Metric ({metric_to_optimize}): {current_val_metric:.4f} | "
                f"Duration: {format_time(epoch_duration)}"
            )

            # Report intermediate value to Optuna for pruning.
            trial.report(current_val_metric, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                logger.warning(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                # Optuna raises TrialPruned; return value is used internally by Optuna.
                raise optuna.TrialPruned()

            # Update the best metric *for this specific trial*
            if optimization_direction == "maximize":
                best_val_metric_for_trial = max(
                    best_val_metric_for_trial, current_val_metric
                )
            else:  # minimize
                best_val_metric_for_trial = min(
                    best_val_metric_for_trial, current_val_metric
                )

        # --- Trial Completion ---
        total_trial_time = time.time() - trial_start_time
        # <<< ADDED: Log trial end info to main logger >>>
        logger.info(
            f"Trial {trial.number} finished. Duration: {format_time(total_trial_time)}. "
            f"Best Value ({metric_to_optimize}): {best_val_metric_for_trial:.4f}"
        )
        # --- END ADDED Logging ---

        # Return the final best metric achieved in this trial
        return best_val_metric_for_trial

    except optuna.TrialPruned as e:
        # Let Optuna handle the pruned state.
        raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        # Return a very poor value to indicate failure.
        # Check direction to return appropriate worst value.
        return -float("inf") if optimization_direction == "maximize" else float("inf")
    finally:
        # --- Cleanup ---
        # Ensure resources are released, especially GPU memory
        del model, optimizer, train_loader, val_loader, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} ---")