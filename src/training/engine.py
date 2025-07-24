# File: ./src/training/engine.py
"""Core training and evaluation engine for experiments."""

import contextlib
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from src.algorithms import (
    get_evaluation_function,
    get_training_function,
)
from src.architectures import FF_MLP, MF_MLP, CaFo_CNN
from src.data_utils.datasets import get_dataloaders
from src.utils.codecarbon_utils import setup_codecarbon_tracker
from src.utils.helpers import format_time, set_seed
from src.utils.logging_utils import log_metrics, logger, setup_wandb
from src.utils.monitoring import (
    GPUEnergyMonitor,
    get_gpu_handle,
    get_gpu_memory_usage,
    init_nvml,
    shutdown_nvml,
)
from src.utils.profiling import profile_model_flops


def _create_mlp_model(
    arch_name: str,
    is_bp_baseline: bool,
    config: Dict[str, Any],
    device: torch.device,
    arch_params: Dict[str, Any],
    num_classes: int,
) -> nn.Module:
    """Creates an MLP-based model (FF_MLP, MF_MLP) or its BP baseline."""
    model: Optional[nn.Module] = None
    if arch_name == "ff_mlp":
        if is_bp_baseline:
            logger.info(
                "Adapting FF_MLP structure for BP baseline -> nn.Sequential MLP."
            )
            bp_input_dim = arch_params["input_dim"]
            hidden_dims = arch_params.get("hidden_dims", [])
            activation_name = arch_params.get("activation", "ReLU").lower()
            use_bias = arch_params.get("bias", True)
            if not hidden_dims:
                raise ValueError(
                    "BP baseline creation failed: hidden_dims missing for FF_MLP."
                )
            layers = []
            current_dim = bp_input_dim
            act_cls = nn.ReLU if activation_name == "relu" else nn.Tanh
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
                layers.append(act_cls())
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
            model = nn.Sequential(*layers)
        else:
            model = FF_MLP(config=config, device=device, **arch_params)
            logger.debug("Using native modified FF_MLP structure.")
    elif arch_name == "mf_mlp":
        model = MF_MLP(**arch_params)
        logger.info(
            "Using MF_MLP for %s", "BP baseline" if is_bp_baseline else "native MF."
        )
    return model


def _create_cnn_model(
    is_bp_baseline: bool,
    device: torch.device,
    arch_params: Dict[str, Any],
    num_classes: int,
) -> nn.Module:
    """Creates a CNN-based model (CaFo_CNN) or its BP baseline."""
    cnn_base = CaFo_CNN(**arch_params)
    if not is_bp_baseline:
        logger.debug("Using native CaFo_CNN structure.")
        return cnn_base

    logger.info("Creating BP baseline from CaFo_CNN blocks + final Linear layer.")
    cnn_base.to(device)
    with torch.no_grad():
        dummy_input = torch.randn(
            1,
            arch_params["input_channels"],
            arch_params["image_size"],
            arch_params["image_size"],
        ).to(device)
        dummy_features = dummy_input
        for block in cnn_base.blocks:
            dummy_features = block(dummy_features)
        num_output_features = dummy_features.view(dummy_features.size(0), -1).shape[1]
        logger.debug(
            f"Dynamically computed flattened output dimension: {num_output_features}"
        )
    cnn_base.cpu()

    model = nn.Sequential(
        *cnn_base.blocks,
        nn.Flatten(),
        nn.Linear(num_output_features, num_classes),
    )
    logger.debug("Created BP baseline Sequential model from CaFo_CNN spec.")
    return model


def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, Optional[Callable[[torch.Tensor], torch.Tensor]]]:
    """Instantiates a model and an optional input adapter based on config."""
    model_config = config.get("model", {})
    arch_name = model_config.get("name", "").lower()
    arch_params = model_config.get("params", {})
    dataset_config = config.get("data", {})
    num_classes = dataset_config.get("num_classes", 10)
    input_channels = dataset_config.get("input_channels", 1)
    image_size = dataset_config.get("image_size", 28)
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    logger.info(f"Getting architecture: {arch_name} (Algo: {algorithm_name.upper()})")
    if "num_classes" not in arch_params:
        arch_params["num_classes"] = num_classes

    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    if arch_name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
        input_adapter = lambda x: x.view(x.shape[0], -1)  # noqa: E731
    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
    else:
        raise ValueError(f"Unknown model architecture category: {arch_name}.")

    if arch_name in ["ff_mlp", "mf_mlp"]:
        model = _create_mlp_model(
            arch_name, is_bp_baseline, config, device, arch_params, num_classes
        )
    elif arch_name == "cafo_cnn":
        model = _create_cnn_model(is_bp_baseline, device, arch_params, num_classes)
    else:
        # This case is already handled above, but as a safeguard:
        raise ValueError(f"Unhandled architecture type: {arch_name}")

    if model is None:
        raise RuntimeError(f"Model instantiation failed for architecture: {arch_name}")

    logger.info(
        f"Model '{arch_name.upper()}' (Algo: {algorithm_name.upper()}) created."
    )
    return model, input_adapter


def _setup_environment_and_wandb(
    config: Dict[str, Any], wandb_run: Optional["wandb.sdk.wandb_run.Run"]
) -> Tuple[int, torch.device, Optional["wandb.sdk.wandb_run.Run"]]:
    """Initializes seed, device, and Weights & Biases."""
    general_config = config.get("general", {})
    seed = general_config.get("seed", 42)
    env_seed_str = os.environ.get("EXPERIMENT_SEED")
    if env_seed_str:
        try:
            seed = int(env_seed_str)
            logger.info(f"Overriding config seed with EXPERIMENT_SEED={seed}.")
            general_config["seed"] = seed
        except ValueError:
            logger.warning(
                f"Invalid EXPERIMENT_SEED: '{env_seed_str}'. Using config seed."
            )
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")

    device_pref = general_config.get("device", "auto").lower()
    device = torch.device(
        "cuda"
        if device_pref == "cuda" and torch.cuda.is_available()
        else (
            "cpu"
            if device_pref == "cpu"
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    logger.info(f"Using device: {device} (Preference: '{device_pref}')")

    if wandb_run is None:
        wandb_run = setup_wandb(config, job_type="training")
    if wandb_run:
        try:
            wandb_run.define_metric("global_step", summary="max")
            wandb_run.define_metric("*", step_metric="global_step")
        except Exception as e:
            logger.error(f"Failed to define W&B metrics: {e}")

    return seed, device, wandb_run


def _setup_hardware_monitors(
    config: Dict[str, Any], device: torch.device, results: Dict[str, Any]
) -> Tuple[
    bool, Optional[Any], Optional[GPUEnergyMonitor], Optional[Any], Optional[str], Dict
]:
    """Initializes NVML, GPUEnergyMonitor, and CodeCarbon."""
    initial_metrics = {}
    monitoring_config = config.get("monitoring", {})
    nvml_active, gpu_handle, monitor = False, None, None

    if device.type == "cuda" and init_nvml():
        nvml_active = True
        gpu_index = torch.cuda.current_device()
        gpu_handle = get_gpu_handle(gpu_index)
        if gpu_handle:
            logger.info(f"NVML active for GPU {gpu_index}.")
            mem_info = get_gpu_memory_usage(gpu_handle)
            if mem_info:
                initial_metrics["initial_gpu_mem_used_mib"] = mem_info[0]
                logger.info(
                    f"Initial GPU Mem: {mem_info[0]:.2f} / {mem_info[1]:.2f} MiB"
                )
            if monitoring_config.get("energy_enabled", True):
                monitor = GPUEnergyMonitor(device_index=gpu_index)
                logger.info("GPU Energy monitor initialized.")
        else:
            nvml_active = False

    tracker = setup_codecarbon_tracker(config, results)
    carbon_csv_path = results.get("codecarbon_csv_path")
    for key in ["codecarbon_enabled", "codecarbon_mode", "codecarbon_country_iso"]:
        initial_metrics[key] = results.get(key, "N/A")

    return nvml_active, gpu_handle, monitor, tracker, carbon_csv_path, initial_metrics


def _perform_pre_training_profiling(
    config: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    input_adapter: Optional[Callable],
) -> Dict[str, Any]:
    """Profiles model parameters and FLOPs."""
    profile_metrics = {}
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    profile_metrics["model_parameters_trainable"] = num_params
    profile_metrics["model_parameters_total"] = sum(
        p.numel() for p in model.parameters()
    )
    logger.info(
        f"Model parameters (Trainable / Total): {num_params:,} / {profile_metrics['model_parameters_total']:,}"
    )

    profiling_config = config.get("profiling", {})
    if not profiling_config.get("enabled", True):
        return profile_metrics

    try:
        sample_input, _ = next(iter(train_loader))
        sample_input_device = sample_input.to(device)

        def input_constructor() -> torch.Tensor:
            if input_adapter:
                return input_adapter(sample_input_device[:1])
            return sample_input_device[:1]

        logger.info("Profiling FLOPs...")
        gflops = profile_model_flops(
            model, input_constructor, device, profiling_config.get("verbose", False)
        )
        profile_metrics["estimated_fwd_gflops"] = gflops or float("nan")
        if gflops:
            logger.info(f"Estimated Forward Pass GFLOPs: {gflops:.4f} G")
            if config.get("algorithm", {}).get("name", "").lower() == "bp":
                bp_gflops = gflops * 3.0
                profile_metrics["estimated_bp_update_gflops"] = bp_gflops
                logger.info(f"Estimated BP Update GFLOPs: {bp_gflops:.4f} G")
        else:
            logger.warning("FLOPs profiling failed.")
    except StopIteration:
        logger.warning("Empty DataLoader, cannot perform FLOPs profiling.")
    except Exception as e:
        logger.error(f"FLOPs profiling failed: {e}", exc_info=True)

    return profile_metrics


def _read_codecarbon_emissions(carbon_csv_path: str) -> Tuple[float, float]:
    """Reads final emissions from CodeCarbon CSV with retries."""
    if not carbon_csv_path or not os.path.exists(carbon_csv_path):
        if carbon_csv_path:
            logger.warning(f"CodeCarbon CSV path '{carbon_csv_path}' does not exist.")
        return float("nan"), float("nan")

    for attempt in range(5):
        try:
            time.sleep(0.5)
            df = pd.read_csv(carbon_csv_path)
            if (
                not df.empty
                and "emissions" in df.columns
                and pd.notna(df["emissions"].iloc[-1])
            ):
                emissions_kg = float(df["emissions"].iloc[-1])
                emissions_g = emissions_kg * 1000.0
                logger.info(
                    f"Read emissions: {emissions_kg:.6f} kgCO2e ({emissions_g:.3f} gCO2e)"
                )
                return emissions_kg, emissions_g
        except (pd.errors.EmptyDataError, FileNotFoundError, Exception) as e_read:
            logger.warning(f"Read CSV attempt {attempt + 1} failed: {e_read}")
    logger.error(
        f"Failed to read valid emissions from {carbon_csv_path} after retries."
    )
    return float("nan"), float("nan")


def _finalize_run(
    run_start_time: float,
    results: Dict[str, Any],
    step_ref: List[int],
    tracker: Optional[Any],
    carbon_csv_path: Optional[str],
    monitor: Optional[GPUEnergyMonitor],
    nvml_active: bool,
    gpu_handle: Optional[Any],
    wandb_run: Optional["wandb.sdk.wandb_run.Run"],
) -> None:
    """Stops monitors, cleans up resources, and logs final summary metrics."""
    total_run_time = time.time() - run_start_time
    results["total_run_duration_sec"] = total_run_time

    if tracker:
        logger.info("Stopping CodeCarbon tracker...")
        tracker.stop()
        _, emissions_g = _read_codecarbon_emissions(carbon_csv_path)
        results["codecarbon_emissions_gCO2e"] = emissions_g

    total_energy_joules = monitor.stop() if monitor else None
    if total_energy_joules is not None:
        results["total_gpu_energy_joules"] = total_energy_joules
        results["total_gpu_energy_wh"] = total_energy_joules / 3600.0

    if nvml_active and gpu_handle and (mem_info := get_gpu_memory_usage(gpu_handle)):
        logger.info(f"GPU Mem (End): {mem_info[0]:.2f} / {mem_info[1]:.2f} MiB")

    final_summary_metrics = {
        "global_step": step_ref[0] + 1,
        "final/total_run_duration_sec": total_run_time,
        "final/training_duration_sec": results.get(
            "training_duration_sec", float("nan")
        ),
        "final/Test_Accuracy": results.get("test_accuracy", float("nan")),
        "final/Test_Loss": results.get("test_loss", float("nan")),
        "final/peak_gpu_mem_used_mib": results.get(
            "peak_gpu_mem_used_mib", float("nan")
        ),
        "final/total_gpu_energy_joules": results.get(
            "total_gpu_energy_joules", float("nan")
        ),
        "final/total_gpu_energy_wh": results.get("total_gpu_energy_wh", float("nan")),
        "final/estimated_fwd_gflops": results.get("estimated_fwd_gflops", float("nan")),
        "final/estimated_bp_update_gflops": results.get(
            "estimated_bp_update_gflops", float("nan")
        ),
        "final/codecarbon_emissions_gCO2e": results.get(
            "codecarbon_emissions_gCO2e", float("nan")
        ),
    }
    log_metrics(final_summary_metrics, wandb_run=wandb_run, commit=True)
    logger.info(f"Total run duration: {format_time(total_run_time)}")
    if pd.notna(results.get("codecarbon_emissions_gCO2e")):
        logger.info(
            f"--> Final Emissions: {results['codecarbon_emissions_gCO2e']:.3f} gCO2e"
        )

    if wandb_run and results.get("error") is None:
        wandb_run.finish()
    if nvml_active:
        shutdown_nvml()


def run_training(
    config: Dict[str, Any], wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
) -> Dict[str, Any]:
    """Orchestrates the entire training and evaluation pipeline for an experiment."""
    results: Dict[str, Any] = {}
    run_start_time = time.time()
    step_ref = [-1]

    try:
        seed, device, wandb_run = _setup_environment_and_wandb(config, wandb_run)

        (
            nvml_active,
            gpu_handle,
            monitor,
            tracker,
            carbon_csv_path,
            initial_metrics,
        ) = _setup_hardware_monitors(config, device, results)

        data_config = config.get("data", {})
        loader_config = config.get("data_loader", {})
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=seed,
            num_workers=loader_config.get("num_workers", 4),
            pin_memory=(
                loader_config.get("pin_memory", True) and device.type == "cuda"
            ),
        )
        logger.info("Dataloaders created.")

        model, input_adapter = get_model_and_adapter(config, device)
        model.to(device)

        profile_metrics = _perform_pre_training_profiling(
            config, model, train_loader, device, input_adapter
        )
        initial_metrics.update(profile_metrics)

        step_ref[0] = 0
        log_metrics(
            {"global_step": step_ref[0], **initial_metrics},
            wandb_run=wandb_run,
            commit=True,
        )

        logger.info("Starting training phase...")
        train_loop_start_time = time.time()
        algo_name = config.get("algorithm", {}).get("name", "").lower()
        training_fn = get_training_function(algo_name)
        with monitor if monitor else contextlib.nullcontext():
            train_args = {
                "model": model,
                "train_loader": train_loader,
                "config": config,
                "device": device,
                "wandb_run": wandb_run,
                "step_ref": step_ref,
                "gpu_handle": gpu_handle,
                "nvml_active": nvml_active,
                "val_loader": val_loader,
                "input_adapter": input_adapter,
            }
            train_output = training_fn(**train_args)
            results["peak_gpu_mem_used_mib"] = (
                train_output if isinstance(train_output, (float, int)) else float("nan")
            )
        results["training_duration_sec"] = time.time() - train_loop_start_time
        logger.info(
            f"Training finished in {format_time(results['training_duration_sec'])}."
        )

        logger.info("Starting evaluation phase on test set...")
        evaluation_fn = get_evaluation_function(algo_name)
        eval_criterion = nn.CrossEntropyLoss()
        eval_args = {
            "model": model,
            "data_loader": test_loader,
            "device": device,
            "criterion": eval_criterion,
            "input_adapter": input_adapter,
        }
        if algo_name == "cafo":
            eval_args["aggregation_method"] = config.get("algorithm_params", {}).get(
                "aggregation_method", "sum"
            )
        eval_output = evaluation_fn(**eval_args)
        if isinstance(eval_output, dict):
            results["test_loss"] = eval_output.get("eval_loss", float("nan"))
            results["test_accuracy"] = eval_output.get("eval_accuracy", float("nan"))
        elif isinstance(eval_output, tuple) and len(eval_output) == 2:
            results["test_loss"], results["test_accuracy"] = eval_output
        logger.info(
            f"Test Set Results: Acc: {results.get('test_accuracy', 'N/A'):.2f}%, "
            f"Loss: {results.get('test_loss', 'N/A'):.4f}"
        )

    except Exception as e:
        logger.critical(f"Experiment Failed: {e}", exc_info=True)
        results["error"] = str(e)
        if wandb_run:
            wandb_run.finish(exit_code=1)
    finally:
        _finalize_run(
            run_start_time,
            results,
            step_ref,
            tracker,
            carbon_csv_path,
            monitor,
            nvml_active,
            gpu_handle,
            wandb_run,
        )

    return results
