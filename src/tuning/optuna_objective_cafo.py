"""Optuna objective function for Cascaded Forward (CaFo) model tuning."""

import copy
import logging
import pprint
import time
from typing import Any, Dict, Tuple

import optuna
import torch
import torch.nn as nn

from src.algorithms.cafo import evaluate_cafo_model, train_cafo_model
from src.data_utils.datasets import get_dataloaders
from src.training.engine import get_model_and_adapter
from src.utils.helpers import format_time, set_seed

logger = logging.getLogger(__name__)


def _setup_cafo_trial(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], torch.device, int]:
    """Suggests hyperparameters and sets up the environment for a CaFo trial."""
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for CaFo tuning.")

    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    # Suggest hyperparameters
    lr_range_fallback = tuning_cfg.get("lr_range", [1e-5, 1e-2])
    wd_range_fallback = tuning_cfg.get("wd_range", [1e-6, 1e-3])
    pred_lr_range = tuning_cfg.get("cafo_predictor_lr_range", lr_range_fallback)
    epochs_range = tuning_cfg.get("cafo_epochs_per_block_range", [10, 200])
    pred_wd_range = tuning_cfg.get("cafo_predictor_wd_range", wd_range_fallback)

    cfg["algorithm_params"]["predictor_lr"] = trial.suggest_float(
        "pred_lr", *pred_lr_range, log=True
    )
    cfg["algorithm_params"]["num_epochs_per_block"] = trial.suggest_int(
        "epochs_per_block", *epochs_range
    )
    cfg["algorithm_params"]["predictor_weight_decay"] = trial.suggest_float(
        "pred_wd", *pred_wd_range, log=True
    )

    if cfg.get("algorithm_params", {}).get("train_blocks", False):
        logger.info("Block training enabled, suggesting block hyperparameters.")
        block_lr_range = tuning_cfg.get("cafo_block_lr_range", [1e-6, 1e-3])
        block_wd_range = tuning_cfg.get("cafo_block_wd_range", [1e-7, 1e-4])
        cfg["algorithm_params"]["block_lr"] = trial.suggest_float(
            "block_lr", *block_lr_range, log=True
        )
        cfg["algorithm_params"]["block_weight_decay"] = trial.suggest_float(
            "block_wd", *block_wd_range, log=True
        )

    # Setup environment
    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return cfg, device, trial_seed


def objective_cafo(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter tuning of Cascaded Forward (CaFo)."""
    cfg, device, trial_seed = _setup_cafo_trial(trial, base_config)
    tuning_cfg = cfg["tuning"]

    logger.info(
        f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) for CaFo ---"
    )
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    logger.info(f"  CaFo Hyperparameters:\n{pprint.pformat(trial.params)}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy":
        logger.warning(
            f"CaFo optimization metric '{metric_to_optimize}' is not 'val_accuracy'."
        )

    model = None
    try:
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "CIFAR10"),
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

        model, input_adapter = get_model_and_adapter(cfg, device)
        model.to(device)
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' created."
        )

        trial_train_start_time = time.time()
        train_cafo_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            device=device,
            wandb_run=None,
            input_adapter=None,  # CaFo handles its own data internally
            step_ref=[-1],
            gpu_handle=None,
            nvml_active=False,
        )
        logger.info(
            f"CaFo training completed in {format_time(time.time() - trial_train_start_time)}."
        )

        aggregation_method = cfg.get("algorithm_params", {}).get(
            "aggregation_method", "sum"
        )
        eval_criterion = (
            nn.CrossEntropyLoss() if metric_to_optimize == "val_loss" else None
        )
        eval_results = evaluate_cafo_model(
            model=model,
            data_loader=val_loader,
            device=device,
            aggregation_method=aggregation_method,
            criterion=eval_criterion,
        )
        metric_key = f"eval_{metric_to_optimize.replace('val_', '')}"
        final_metric_value = eval_results.get(metric_key, float("nan"))

        if torch.isnan(torch.tensor(final_metric_value)):
            raise ValueError(f"Evaluation returned NaN {metric_to_optimize}.")

        logger.info(
            f"Trial {trial.number} finished. Final Metric ({metric_to_optimize}): {final_metric_value:.4f}"
        )
        return final_metric_value

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
        logger.info(f"--- Finished Optuna Trial {trial.number} for CaFo ---")
