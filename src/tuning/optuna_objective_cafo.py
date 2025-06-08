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
from src.algorithms.cafo import train_cafo_model, evaluate_cafo_model

logger = logging.getLogger(__name__)

def objective_cafo(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter tuning of Cascaded Forward (CaFo).
    Focuses on tuning predictor learning rate and epochs per block.
    Can be extended to tune block training parameters if train_blocks is enabled.
    """
    cfg = copy.deepcopy(base_config)
    tuning_cfg = cfg.get("tuning")
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Missing 'tuning' section in configuration for CaFo tuning.")

    if "algorithm_params" not in cfg or not isinstance(cfg["algorithm_params"], dict):
        cfg["algorithm_params"] = {}

    pred_lr_range = tuning_cfg.get("cafo_predictor_lr_range", tuning_cfg.get("lr_range", [1e-5, 1e-2]))
    epochs_range = tuning_cfg.get("cafo_epochs_per_block_range", [10, 200])
    pred_wd_range = tuning_cfg.get("cafo_predictor_wd_range", tuning_cfg.get("wd_range", [1e-6, 1e-3]))

    cfg["algorithm_params"]["predictor_lr"] = trial.suggest_float(
        "pred_lr", *pred_lr_range, log=True
    )
    cfg["algorithm_params"]["num_epochs_per_block"] = trial.suggest_int(
        "epochs_per_block", *epochs_range
    )
    cfg["algorithm_params"]["predictor_weight_decay"] = trial.suggest_float(
        "pred_wd", *pred_wd_range, log=True
    )

    train_blocks_flag = cfg.get("algorithm_params", {}).get("train_blocks", False)
    if train_blocks_flag:
        logger.info("Block training enabled, suggesting block hyperparameters.")
        block_lr_range = tuning_cfg.get("cafo_block_lr_range", [1e-6, 1e-3])
        block_wd_range = tuning_cfg.get("cafo_block_wd_range", [1e-7, 1e-4])

        cfg["algorithm_params"]["block_lr"] = trial.suggest_float(
            "block_lr", *block_lr_range, log=True
        )
        cfg["algorithm_params"]["block_weight_decay"] = trial.suggest_float(
            "block_wd", *block_wd_range, log=True
        )


    trial_seed = cfg.get("general", {}).get("seed", 42) + trial.number
    set_seed(trial_seed)
    device_name = cfg.get("general", {}).get("device", "auto").lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"--- Starting Optuna Trial {trial.number} (Study: {trial.study.study_name}) for CaFo ---")
    logger.info(f"  Device: {device}, Seed: {trial_seed}")
    param_str = pprint.pformat(trial.params)
    logger.info(f"  CaFo Hyperparameters:\n{param_str}")

    metric_to_optimize = tuning_cfg.get("metric", "val_accuracy").lower()
    optimization_direction = tuning_cfg.get("direction", "maximize").lower()
    if metric_to_optimize != "val_accuracy":
        logger.warning(f"CaFo optimization metric '{metric_to_optimize}' is not 'val_accuracy'.")

    model = None
    train_loader = None
    val_loader = None

    try:
        data_config = cfg.get("data", {})
        loader_config = cfg.get("data_loader", {})
        logger.info(f"Trial {trial.number}: Loading data...")
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

        logger.info(f"Trial {trial.number}: Instantiating CaFo model...")
        model, input_adapter = get_model_and_adapter(cfg, device)
        model.to(device)
        num_block_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trial {trial.number}: Model '{cfg.get('model', {}).get('name')}' blocks ({num_block_params:,} params) on {device}."
        )
        if input_adapter is not None:
             logger.warning("CaFo trial: Input adapter was returned but CaFo_CNN typically doesn't use it.")

        logger.info(f"Trial {trial.number}: Starting CaFo training process...")
        trial_train_start_time = time.time()

        minimal_cfg_for_train = copy.deepcopy(cfg)
        minimal_cfg_for_train['monitoring'] = {'enabled': False, 'energy_enabled': False}
        minimal_cfg_for_train['profiling'] = {'enabled': False}
        minimal_cfg_for_train['checkpointing'] = {'checkpoint_dir': None}
        minimal_cfg_for_train['logging'] = {'wandb': {'use_wandb': False}}

        step_ref = [-1]
        _ = train_cafo_model(
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
        logger.info(f"Trial {trial.number}: CaFo training completed in {format_time(trial_train_duration)}.")

        logger.info(f"Trial {trial.number}: Evaluating CaFo model on validation set...")
        eval_results = evaluate_cafo_model(
            model=model,
            data_loader=val_loader,
            device=device,
            aggregation_method=cfg.get("algorithm_params",{}).get("aggregation_method", "sum"),
            criterion=nn.CrossEntropyLoss() if metric_to_optimize == "val_loss" else None
        )
        final_metric_value = eval_results.get(f"eval_{metric_to_optimize.replace('val_', '')}", float('nan'))

        if metric_to_optimize == "val_accuracy":
             logger.info(f"Trial {trial.number}: Validation Accuracy: {final_metric_value:.2f}%")
        else:
             logger.info(f"Trial {trial.number}: Validation Loss: {final_metric_value:.4f}")


        if torch.isnan(torch.tensor(final_metric_value)):
             logger.error(f"Trial {trial.number}: Evaluation returned NaN {metric_to_optimize}. Treating as failure.")
             raise ValueError("Evaluation failed.")

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
        del model, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug(f"Trial {trial.number}: Cleared CUDA cache.")
        logger.info(f"--- Finished Optuna Trial {trial.number} for CaFo ---")