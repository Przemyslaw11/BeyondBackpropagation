# File: ./src/tuning/optuna_objective_mf.py
import optuna
import torch
import torch.nn as nn
import logging
import time
import copy
import pprint
from typing import Dict, Any, Optional, Tuple, Callable, List

from src.utils.helpers import set_seed, format_time
from src.data_utils.datasets import get_dataloaders
from src.training.engine import get_model_and_adapter
from src.algorithms.mf import train_mf_model, evaluate_mf_model # Import MF specific functions

# Use the centrally configured logger
logger = logging.getLogger(__name__)

def objective_mf(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter tuning of Mono-Forward (MF).
    NOTE: Early stopping (if enabled in config) will happen *within* the
          train_mf_model call during the trial, potentially reducing the
          number of epochs trained per layer. Optuna itself does not prune
          based on intermediate layer results in this setup.
    """
    # --- Hyperparameter Suggestion & Config Setup ---
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for MF tuning.")

    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    lr_range = tuning_cfg.get("mf_lr_range", tuning_cfg.get("lr_range", [1e-5, 1e-2]))
    epochs_range = tuning_cfg.get("mf_epochs_per_layer_range", [5, 50])

    cfg["algorithm_params"]["lr"] = trial.suggest_float("lr", *lr_range, log=True)
    cfg["algorithm_params"]["epochs_per_layer"] = trial.suggest_int("epochs_per_layer", *epochs_range)
    # Optionally tune WD:
    # wd_range = tuning_cfg.get("mf_wd_range", tuning_cfg.get("wd_range", [1e-6, 1e-3]))
    # cfg["algorithm_params"]["weight_decay"] = trial.suggest_float("wd", *wd_range, log=True)

    # --- Setup (Seed, Device) ---
    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")

    logger.info(f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) for MF ---")
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    param_str = pprint.pformat(trial.params); logger.info(f"  MF Hyperparameters:\n{param_str}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy":
        logger.warning(f"MF optimization metric '{metric_to_optimize}' is not 'val_accuracy'.")

    model = None; train_loader = None; val_loader = None

    try:
        # --- Data Loading ---
        data_config = cfg.get("data", {}); loader_config = cfg.get("data_loader", {})
        logger.info(f"Trial {trial.number}: Loading data...")
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"), batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"), val_split=data_config.get("val_split", 0.1),
            seed=trial_seed, num_workers=0, pin_memory=False, download=data_config.get("download", True),
        )
        if not val_loader and cfg.get("algorithm_params",{}).get("mf_early_stopping_enabled", False):
             logger.warning(f"Trial {trial.number}: MF Early stopping is enabled but val_loader is None. Disabling ES for this trial.")
             cfg["algorithm_params"]["mf_early_stopping_enabled"] = False
        elif not val_loader:
             logger.info(f"Trial {trial.number}: No validation loader, MF early stopping cannot be used.")
        logger.info(f"Trial {trial.number}: Data loaded.")

        # --- Model Instantiation & Adapter ---
        logger.info(f"Trial {trial.number}: Instantiating MF model and getting adapter...")
        model, input_adapter = get_model_and_adapter(cfg, device); model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' ({num_params:,} params) on {device}.")

        # --- MF Training ---
        logger.info(f"Trial {trial.number}: Starting MF layer-wise training...")
        trial_train_start_time = time.time()

        minimal_cfg_for_train = copy.deepcopy(cfg)
        minimal_cfg_for_train['monitoring'] = {'enabled': False, 'energy_enabled': False}
        minimal_cfg_for_train['profiling'] = {'enabled': False}
        minimal_cfg_for_train['checkpointing'] = {'checkpoint_dir': None}
        minimal_cfg_for_train['logging'] = {'wandb': {'use_wandb': False}} # Disable W&B for trial

        step_ref = [-1]

        # --- Pass val_loader to train_mf_model --- #
        _ = train_mf_model(
            model=model, train_loader=train_loader, val_loader=val_loader, # Pass val_loader here
            config=minimal_cfg_for_train, device=device, wandb_run=None,
            input_adapter=input_adapter, step_ref=step_ref, gpu_handle=None, nvml_active=False,
        )
        trial_train_duration = time.time() - trial_train_start_time
        logger.info(f"Trial {trial.number}: MF training completed in {format_time(trial_train_duration)}.")

        # --- MF Evaluation (on Validation Set) ---
        logger.info(f"Trial {trial.number}: Evaluating MF model on validation set...")
        eval_results = evaluate_mf_model(model=model, data_loader=val_loader, device=device, input_adapter=input_adapter)
        validation_accuracy = eval_results.get("eval_accuracy", float("nan"))
        logger.info(f"Trial {trial.number}: Validation Accuracy: {validation_accuracy:.2f}%")

        if torch.isnan(torch.tensor(validation_accuracy)):
             logger.error(f"Trial {trial.number}: Evaluation returned NaN accuracy."); raise ValueError("Eval failed.")

        # --- Trial Completion ---
        logger.info(f"Trial {trial.number} finished. Final Validation Accuracy: {validation_accuracy:.4f}")
        return validation_accuracy # Return final accuracy

    except optuna.TrialPruned as e: raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        return -1.0 # Poor value for maximization
    finally:
        del model, train_loader, val_loader
        if device.type == "cuda": torch.cuda.empty_cache(); logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} for MF ---")