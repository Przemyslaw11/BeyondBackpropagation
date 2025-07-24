"""Optuna objective function for Forward-Forward (FF) model tuning."""

import copy
import logging
import pprint
import time
from typing import Any, Dict, Tuple

import optuna
import torch

from src.algorithms.ff import evaluate_ff_model, train_ff_model
from src.architectures.ff_mlp import FF_MLP
from src.data_utils.datasets import get_dataloaders
from src.utils.helpers import format_time, set_seed

logger = logging.getLogger(__name__)


def _setup_ff_trial(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], torch.device, int]:
    """Suggests hyperparameters and sets up the environment for an FF trial."""
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for FF tuning.")

    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    # Suggest hyperparameters
    cfg["algorithm_params"]["ff_learning_rate"] = trial.suggest_float(
        "ff_lr", *tuning_cfg.get("ff_lr_range", [1e-5, 1e-2]), log=True
    )
    cfg["algorithm_params"]["ff_weight_decay"] = trial.suggest_float(
        "ff_wd", *tuning_cfg.get("ff_wd_range", [1e-6, 1e-3]), log=True
    )
    cfg["algorithm_params"]["downstream_learning_rate"] = trial.suggest_float(
        "ds_lr", *tuning_cfg.get("ds_lr_range", [1e-4, 1e-1]), log=True
    )
    cfg["algorithm_params"]["downstream_weight_decay"] = trial.suggest_float(
        "ds_wd", *tuning_cfg.get("ds_wd_range", [1e-5, 1e-2]), log=True
    )
    cfg["training"]["epochs"] = tuning_cfg.get("num_epochs", 50)

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


def objective_ff(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter tuning of Forward-Forward (FF)."""
    cfg, device, trial_seed = _setup_ff_trial(trial, base_config)
    tuning_cfg = cfg["tuning"]

    logger.info(
        f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) for FF ---"
    )
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    logger.info(f"  FF Hyperparameters:\n{pprint.pformat(trial.params)}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy":
        logger.warning("FF eval only returns accuracy. Optimizing 'val_accuracy'.")
        metric_to_optimize = "val_accuracy"

    model = None
    try:
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=data_config.get("name", "MNIST"),
            batch_size=loader_config.get("batch_size", 100),
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

        model = FF_MLP(config=cfg, device=device)
        model.to(device)
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' created."
        )

        trial_train_start_time = time.time()
        train_ff_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            device=device,
            wandb_run=None,
            input_adapter=None,
            step_ref=[-1],
            gpu_handle=None,
            nvml_active=False,
        )
        logger.info(
            f"FF training completed in {format_time(time.time() - trial_train_start_time)}."
        )

        eval_results = evaluate_ff_model(
            model=model, data_loader=val_loader, device=device
        )
        validation_accuracy = eval_results.get("eval_accuracy", float("nan"))
        logger.info(
            f"Trial {trial.number}: Validation Accuracy: {validation_accuracy:.2f}%"
        )

        if torch.isnan(torch.tensor(validation_accuracy)):
            raise ValueError("Evaluation returned NaN accuracy.")

        logger.info(
            f"Trial {trial.number} finished. Final Metric: {validation_accuracy:.4f}"
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
        logger.info(f"--- Finished Optuna Trial {trial.number} for FF ---")
