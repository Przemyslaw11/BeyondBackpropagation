# File: src/tuning/optuna_objective_mf.py
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
    """
    # --- Hyperparameter Suggestion & Config Setup ---
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for MF tuning.")

    # Ensure algorithm_params sub-dict exists (MF uses this for lr, epochs_per_layer)
    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    # --- Suggest MF Hyperparameters ---
    # Use specific keys in tuning config if needed, e.g., mf_lr_range
    lr_range = tuning_cfg.get("mf_lr_range", tuning_cfg.get("lr_range", [1e-5, 1e-2]))
    epochs_range = tuning_cfg.get("mf_epochs_per_layer_range", [5, 50]) # Default range 5-50 epochs/layer

    cfg["algorithm_params"]["lr"] = trial.suggest_float(
        "lr", *lr_range, log=True
    )
    cfg["algorithm_params"]["epochs_per_layer"] = trial.suggest_int(
        "epochs_per_layer", *epochs_range
    )
    # Optionally tune weight decay if desired
    # wd_range = tuning_cfg.get("mf_wd_range", tuning_cfg.get("wd_range", [1e-6, 1e-3]))
    # cfg["algorithm_params"]["weight_decay"] = trial.suggest_float(
    #     "wd", *wd_range, log=True
    # )

    # Note: tuning.num_epochs from BP objective is NOT directly applicable here,
    # as MF trial runs full layer-wise training using suggested epochs_per_layer.

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

    logger.info(f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) for MF ---")
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    param_str = pprint.pformat(trial.params)
    logger.info(f"  MF Hyperparameters:\n{param_str}")

    # Define optimization direction and metric from config
    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy": # MF typically evaluated on accuracy
        logger.warning(
            f"MF optimization metric '{metric_to_optimize}' is not 'val_accuracy'. Ensure evaluation returns this."
        )
        # If you wanted to optimize loss, MF's BP-style eval doesn't return loss.
        # FF-style eval would be needed if loss optimization is the goal.

    model = None  # Initialize for finally block
    train_loader = None
    val_loader = None

    try:
        # --- Data Loading ---
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info(f"Trial {trial.number}: Loading data...")
        # Use smaller batch size or fewer workers for faster trials if needed, but use config defaults first.
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=trial_seed,
            num_workers=0, # Use 0 workers for Optuna trials to simplify debugging
            pin_memory=False,
            download=data_config.get("download", True),
        )
        if not val_loader:
            raise ValueError(
                "Validation loader is required for Optuna tuning but was not created."
            )
        logger.info(f"Trial {trial.number}: Data loaded.")

        # --- Model Instantiation & Input Adapter Retrieval ---
        logger.info(f"Trial {trial.number}: Instantiating MF model and getting adapter...")
        model, input_adapter = get_model_and_adapter(cfg, device) # Should correctly get MF_MLP
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters()) # MF trains all params eventually
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' ({num_params:,} params) on {device}. "
            f"Input adapter type: {type(input_adapter)}"
        )

        # --- MF Training ---
        # MF training happens layer-wise within train_mf_model
        logger.info(f"Trial {trial.number}: Starting MF layer-wise training...")
        trial_train_start_time = time.time()

        # Prepare minimal config for training function if needed, but passing modified cfg should work
        # Ensure monitoring/profiling/checkpointing are off for the trial run itself
        minimal_cfg_for_train = copy.deepcopy(cfg)
        minimal_cfg_for_train['monitoring'] = {'enabled': False, 'energy_enabled': False}
        minimal_cfg_for_train['profiling'] = {'enabled': False}
        minimal_cfg_for_train['checkpointing'] = {'checkpoint_dir': None}

        # Step ref for internal logging steps within train_mf_model (optional)
        step_ref = [-1]

        # Call the main MF training function
        # Disable W&B, monitoring, checkpointing during the trial run itself
        _ = train_mf_model(
            model=model,
            train_loader=train_loader,
            config=minimal_cfg_for_train, # Pass the modified config
            device=device,
            wandb_run=None, # NO W&B LOGGING DURING TRIAL
            input_adapter=input_adapter,
            step_ref=step_ref,
            gpu_handle=None, # No monitoring
            nvml_active=False,
        )
        trial_train_duration = time.time() - trial_train_start_time
        logger.info(f"Trial {trial.number}: MF training completed in {format_time(trial_train_duration)}.")

        # --- MF Evaluation (on Validation Set) ---
        logger.info(f"Trial {trial.number}: Evaluating MF model on validation set...")
        eval_results = evaluate_mf_model( # Use the standard MF evaluation function
            model=model,
            data_loader=val_loader,
            device=device,
            input_adapter=input_adapter,
        )
        validation_accuracy = eval_results.get("eval_accuracy", float("nan"))
        logger.info(f"Trial {trial.number}: Validation Accuracy: {validation_accuracy:.2f}%")

        if torch.isnan(torch.tensor(validation_accuracy)):
             logger.error(f"Trial {trial.number}: Evaluation returned NaN accuracy. Treating as failure.")
             raise ValueError("Evaluation failed.")

        # --- Report Final Metric to Optuna ---
        # MF objective doesn't easily support intermediate pruning like BP epoch loop.
        # We report the final validation accuracy after full layer-wise training.
        # Optuna's pruning mechanism won't be triggered mid-training.
        # trial.report(validation_accuracy, step=cfg["algorithm_params"]["epochs_per_layer"] * (model.num_hidden_layers + 1)) # Report once at the end

        # --- Trial Completion ---
        logger.info(
            f"Trial {trial.number} finished. Final Validation Accuracy ({metric_to_optimize}): {validation_accuracy:.4f}"
        )

        # Return the final metric achieved in this trial
        return validation_accuracy # Assuming optimization_direction='maximize'

    except optuna.TrialPruned as e:
        # Should not happen with current structure, but handle just in case
        logger.warning(f"Trial {trial.number} was pruned unexpectedly.")
        raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        # Return a very poor value to indicate failure.
        return -1.0 # Return very low accuracy for maximization
    finally:
        # --- Cleanup ---
        del model, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} for MF ---")