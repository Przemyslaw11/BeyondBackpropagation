"""Optuna objective function for Mono-Forward (MF) model tuning."""

import copy
import logging
import pprint
import time
from typing import Any, Dict, Tuple

import optuna
import torch

from src.algorithms.mf import evaluate_mf_model, train_mf_model
from src.data_utils.datasets import get_dataloaders
from src.training.engine import get_model_and_adapter
from src.utils.helpers import format_time, set_seed

logger = logging.getLogger(__name__)


def _setup_mf_trial(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], torch.device, int]:
    """Suggests hyperparameters and sets up the environment for an MF trial."""
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for MF tuning.")

    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    # Suggest hyperparameters
    lr_range = tuning_cfg.get("mf_lr_range", tuning_cfg.get("lr_range", [1e-5, 1e-2]))
    epochs_range = tuning_cfg.get("mf_epochs_per_layer_range", [5, 50])
    cfg["algorithm_params"]["lr"] = trial.suggest_float("lr", *lr_range, log=True)
    cfg["algorithm_params"]["epochs_per_layer"] = trial.suggest_int(
        "epochs_per_layer", *epochs_range
    )

    # Setup environment
    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return cfg, device, trial_seed


def objective_mf(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter tuning of Mono-Forward (MF)."""
    cfg, device, trial_seed = _setup_mf_trial(trial, base_config)
    tuning_cfg = cfg["tuning"]

    logger.info(
        f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) for MF ---"
    )
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    logger.info(f"  MF Hyperparameters:\n{pprint.pformat(trial.params)}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy":
        logger.warning(
            f"MF optimization metric '{metric_to_optimize}' is not 'val_accuracy'."
        )

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

        es_enabled = cfg.get("algorithm_params", {}).get(
            "mf_early_stopping_enabled", False
        )
        if not val_loader and es_enabled:
            logger.warning("MF Early stopping enabled but no val_loader. Disabling ES.")
            cfg["algorithm_params"]["mf_early_stopping_enabled"] = False
        logger.info(f"Trial {trial.number}: Data loaded.")

        model, input_adapter = get_model_and_adapter(cfg, device)
        model.to(device)
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' created."
        )

        trial_train_start_time = time.time()
        train_mf_model(
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
            f"MF training completed in {format_time(time.time() - trial_train_start_time)}."
        )

        eval_results = evaluate_mf_model(
            model=model,
            data_loader=val_loader,
            device=device,
            input_adapter=input_adapter,
        )
        validation_accuracy = eval_results.get("eval_accuracy", float("nan"))
        logger.info(
            f"Trial {trial.number}: Validation Accuracy: {validation_accuracy:.2f}%"
        )

        if torch.isnan(torch.tensor(validation_accuracy)):
            raise ValueError("Evaluation returned NaN accuracy.")

        logger.info(
            f"Trial {trial.number} finished. Final Validation Accuracy: {validation_accuracy:.4f}"
        )
        return validation_accuracy

    except optuna.TrialPruned as e:
        raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        return -1.0 if optimization_direction == "maximize" else float("inf")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} for MF ---")
