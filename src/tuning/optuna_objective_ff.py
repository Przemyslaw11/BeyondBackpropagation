"""Optuna objective function for Forward-Forward (FF) model tuning."""

import copy
import logging
import pprint
import time
from typing import Any, Dict

import optuna
import torch

from src.algorithms.ff import evaluate_ff_model, train_ff_model
from src.architectures.ff_mlp import FF_MLP
from src.data_utils.datasets import get_dataloaders
from src.utils.helpers import format_time, set_seed

logger = logging.getLogger(__name__)


def objective_ff(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter tuning of Forward-Forward (FF).

    Focuses on tuning FF layer LR/WD and Downstream LR/WD.
    """
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for FF tuning.")

    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    ff_lr_range = tuning_cfg.get("ff_lr_range", [1e-5, 1e-2])
    ff_wd_range = tuning_cfg.get("ff_wd_range", [1e-6, 1e-3])
    ds_lr_range = tuning_cfg.get("ds_lr_range", [1e-4, 1e-1])
    ds_wd_range = tuning_cfg.get("ds_wd_range", [1e-5, 1e-2])

    cfg["algorithm_params"]["ff_learning_rate"] = trial.suggest_float(
        "ff_lr", *ff_lr_range, log=True
    )
    cfg["algorithm_params"]["ff_weight_decay"] = trial.suggest_float(
        "ff_wd", *ff_wd_range, log=True
    )
    cfg["algorithm_params"]["downstream_learning_rate"] = trial.suggest_float(
        "ds_lr", *ds_lr_range, log=True
    )
    cfg["algorithm_params"]["downstream_weight_decay"] = trial.suggest_float(
        "ds_wd", *ds_wd_range, log=True
    )

    num_epochs_per_trial = tuning_cfg.get("num_epochs", 50)
    cfg["training"]["epochs"] = num_epochs_per_trial

    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"--- Starting Optuna Trial {trial.number} (Study: "
        f"{trial.study.study_name}) for FF ---"
    )
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    param_str = pprint.pformat(trial.params)
    logger.info(f"  FF Hyperparameters:\n{param_str}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy":
        logger.warning(
            f"FF optimization metric '{metric_to_optimize}' is not 'val_accuracy'. "
            "FF eval only returns accuracy."
        )
        metric_to_optimize = "val_accuracy"

    model = None
    train_loader = None
    val_loader = None

    try:
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info(f"Trial {trial.number}: Loading data...")
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
            raise ValueError(
                "Validation loader is required for Optuna tuning (used by FF evaluate)."
            )
        logger.info(f"Trial {trial.number}: Data loaded.")

        logger.info(f"Trial {trial.number}: Instantiating FF model...")
        model = FF_MLP(config=cfg, device=device)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' "
            f"({num_params:,} params) on {device}."
        )

        logger.info(
            f"Trial {trial.number}: Starting FF training for "
            f"{num_epochs_per_trial} epochs..."
        )
        trial_train_start_time = time.time()

        minimal_cfg_for_train = copy.deepcopy(cfg)
        minimal_cfg_for_train["monitoring"] = {
            "enabled": False,
            "energy_enabled": False,
        }
        minimal_cfg_for_train["profiling"] = {"enabled": False}
        minimal_cfg_for_train["checkpointing"] = {"checkpoint_dir": None}
        minimal_cfg_for_train["logging"] = {"wandb": {"use_wandb": False}}

        step_ref = [-1]

        _ = train_ff_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=minimal_cfg_for_train,
            device=device,
            wandb_run=None,
            input_adapter=None,
            step_ref=step_ref,
            gpu_handle=None,
            nvml_active=False,
        )
        trial_train_duration = time.time() - trial_train_start_time
        logger.info(
            f"Trial {trial.number}: FF training completed in "
            f"{format_time(trial_train_duration)}."
        )
        logger.info(f"Trial {trial.number}: Evaluating FF model on validation set...")
        eval_results = evaluate_ff_model(
            model=model,
            data_loader=val_loader,
            device=device,
        )
        validation_accuracy = eval_results.get("eval_accuracy", float("nan"))
        logger.info(
            f"Trial {trial.number}: Validation Accuracy: {validation_accuracy:.2f}%"
        )

        if torch.isnan(torch.tensor(validation_accuracy)):
            logger.error(
                f"Trial {trial.number}: Evaluation returned NaN accuracy. "
                "Treating as failure."
            )
            raise ValueError("Evaluation failed.")

        logger.info(
            f"Trial {trial.number} finished. Final Validation Accuracy "
            f"({metric_to_optimize}): {validation_accuracy:.4f}"
        )

        return validation_accuracy

    except optuna.TrialPruned as e:
        raise e
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        return -1.0 if optimization_direction == "maximize" else float("inf")
    finally:
        del model, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} for FF ---")