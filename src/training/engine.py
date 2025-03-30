# File: src/training/engine.py
import torch
import torch.nn as nn
import logging
import time
import os
import contextlib
from typing import Dict, Any, Optional, Tuple, Callable

from src.utils.config_parser import load_config
from src.utils.helpers import set_seed, create_directory_if_not_exists, format_time
from src.utils.logging_utils import setup_wandb, log_metrics, logger, setup_logging
from src.utils.monitoring import (
    init_nvml,
    shutdown_nvml,
    get_gpu_handle,
    get_gpu_memory_usage,
    GPUEnergyMonitor,
)
from src.utils.profiling import profile_model_flops
from src.data_utils.datasets import get_dataloaders
from src.architectures import FF_MLP, CaFo_CNN, MF_MLP
from src.algorithms import (
    get_training_function,
    get_evaluation_function,
)  # Use factories


def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, Optional[Callable]]:
    """
    Instantiates the model based on the configuration and returns an optional input adapter.
    Handles specific adaptations needed for BP baselines.
    """
    model_config = config.get("model", {})
    arch_name = model_config.get("name", "").lower()
    arch_params = model_config.get("params", {})
    dataset_config = config.get("data", {})
    num_classes = dataset_config.get("num_classes", 10)
    input_channels = dataset_config.get("input_channels", 1)
    image_size = dataset_config.get("image_size", 28)

    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    model: Optional[nn.Module] = None  # Initialize model as Optional
    input_adapter: Optional[Callable] = None  # Initialize adapter

    logger.info(
        f"Creating model architecture: {arch_name} (BP Baseline: {is_bp_baseline})"
    )

    # --- Determine necessary parameters if not provided ---
    if arch_name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        input_adapter = lambda x: x.view(x.shape[0], -1)  # MLPs need flattened input
    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        input_adapter = None  # CNNs usually don't need adapter

    # --- Instantiate based on architecture name ---
    if arch_name == "ff_mlp":
        # Instantiate the base FF_MLP regardless
        ff_model_instance = FF_MLP(**arch_params)
        if is_bp_baseline:
            logger.info(
                "Adapting FF_MLP structure for BP baseline -> Standard nn.Sequential MLP."
            )
            layers = []
            current_dim = arch_params[
                "input_dim"
            ]  # BP baseline uses original image dim
            # Reconstruct sequential layers from FF_MLP structure
            # Input Adapter Layer of FF becomes first Linear layer for BP
            layers.append(
                nn.Linear(
                    current_dim,
                    ff_model_instance.input_adapter.out_features,
                    bias=ff_model_instance.input_adapter.bias is not None,
                )
            )
            layers.append(type(ff_model_instance.first_layer_activation)())
            current_dim = ff_model_instance.input_adapter.out_features
            # Subsequent FF_Layers become Linear + Activation
            for ff_layer in ff_model_instance.layers:
                layers.append(
                    nn.Linear(
                        current_dim,
                        ff_layer.linear.out_features,
                        bias=ff_layer.linear.bias is not None,
                    )
                )
                layers.append(type(ff_layer.activation)())
                current_dim = ff_layer.linear.out_features
            layers.append(
                nn.Linear(current_dim, num_classes)
            )  # Add final BP classifier
            model = nn.Sequential(*layers)
        else:
            model = ff_model_instance  # Use FF_MLP directly
            # Input adapter for FF handled by its train function internally

    elif arch_name == "cafo_cnn":
        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
            logger.info(
                "Creating BP baseline model from CaFo_CNN blocks + final Linear layer."
            )
            # Temporarily move model to device to calculate output dim
            cafo_base.to(device)
            with torch.no_grad():
                dummy_input = torch.randn(1, input_channels, image_size, image_size).to(
                    device
                )
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel()
            cafo_base.cpu()  # Move back
            logger.debug(
                f"Flattened output dimension from CaFo blocks: {num_output_features}"
            )
            model = nn.Sequential(
                cafo_base.blocks,
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes),
            )
        else:
            model = cafo_base  # CaFo training uses the base model

    elif arch_name == "mf_mlp":
        # MF_MLP instance contains W layers, M parameters, and standard forward for BP
        model = MF_MLP(**arch_params)
        if is_bp_baseline:
            logger.info("Using standard forward pass of MF_MLP for BP baseline.")
        # MF training function uses specific forward methods

    else:
        raise ValueError(f"Unknown model architecture name: {arch_name}")

    if model is None:
        raise RuntimeError(f"Failed to instantiate model for architecture: {arch_name}")
    return model, input_adapter


def run_training(
    config: Dict[str, Any], wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Runs the full training and evaluation pipeline for one experiment configuration.
    """
    results = {}
    nvml_active = False
    gpu_handle = None
    monitor = None
    run_start_time = time.time()

    try:
        # --- Setup ---
        general_config = config.get("general", {})
        seed = general_config.get("seed", 42)
        set_seed(seed)
        logger.info(f"Using random seed: {seed}")

        # --- Device Setup (Reads from config) ---
        device_preference = general_config.get("device", "auto").lower()
        if device_preference == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_preference == "cpu":
            device = torch.device("cpu")
        else:  # Includes "auto" or invalid config value
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} (Preference: '{device_preference}')")

        # --- W&B Initialization ---
        if wandb_run is None:
            wandb_run = setup_wandb(config, job_type="training")

        # --- Monitoring Initialization ---
        monitoring_config = config.get("monitoring", {})
        if device.type == "cuda":
            if init_nvml():
                nvml_active = True
                gpu_index = torch.cuda.current_device()  # Get current device index
                gpu_handle = get_gpu_handle(gpu_index)
                if gpu_handle:
                    logger.info(f"NVML active for GPU {gpu_index}.")
                    mem_info_start = get_gpu_memory_usage(gpu_handle)
                    if mem_info_start:
                        log_metrics(
                            {"initial_gpu_mem_used_mib": mem_info_start[0]},
                            wandb_run=wandb_run,
                        )
                    if monitoring_config.get("energy_enabled", True):
                        monitor = GPUEnergyMonitor(
                            device_index=gpu_index,
                            interval_sec=monitoring_config.get(
                                "energy_interval_sec", 0.2
                            ),
                        )
                        logger.info(
                            f"GPU Energy monitor initialized (Interval: {monitor._interval_sec}s)."
                        )
                    else:
                        logger.info("GPU Energy monitoring disabled in configuration.")
                else:
                    logger.warning(
                        f"NVML active but failed to get handle for GPU {gpu_index}. Monitoring disabled."
                    )
                    nvml_active = False
            else:
                logger.warning("NVML initialization failed. GPU monitoring disabled.")
        else:
            logger.info("Running on CPU, GPU monitoring disabled.")

        # --- Data Loading ---
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
                loader_config.get("pin_memory", True)
                if device.type == "cuda"
                else False
            ),
            download=data_config.get("download", True),
        )
        logger.info("Dataloaders created.")

        # --- Model Instantiation ---
        model, input_adapter = get_model_and_adapter(config, device)
        model.to(device)
        logger.info(
            f"Model '{config.get('model', {}).get('name')}' instantiated and moved to {device}."
        )
        try:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_total_params = sum(p.numel() for p in model.parameters())
            log_metrics(
                {
                    "model_parameters_trainable": num_params,
                    "model_parameters_total": num_total_params,
                },
                wandb_run=wandb_run,
            )
            logger.info(f"Model trainable parameters: {num_params:,}")
            logger.info(f"Model total parameters: {num_total_params:,}")
        except Exception as e:
            logger.warning(f"Could not count model parameters: {e}")

        # --- FLOPs Profiling (Optional) ---
        profiling_config = config.get("profiling", {})
        if profiling_config.get("enabled", True):
            try:
                sample_input_img, _ = next(iter(train_loader))
                # Ensure sample input is on the correct device for profiling
                sample_input_device = sample_input_img.to(device)
                if input_adapter:
                    profile_input_constructor = lambda: input_adapter(
                        sample_input_device
                    )
                else:
                    profile_input_constructor = (
                        lambda: sample_input_device
                    )  # Use device input

                logger.info(f"Profiling FLOPs...")
                gmacs = profile_model_flops(
                    model,
                    input_constructor=profile_input_constructor,
                    device=device,
                    verbose=profiling_config.get("verbose", False),
                )
                if gmacs is not None:
                    results["estimated_gmacs"] = gmacs
                    results["estimated_gflops"] = gmacs * 2.0
                    log_metrics(
                        {
                            "estimated_gmacs": gmacs,
                            "estimated_gflops": results["estimated_gflops"],
                        },
                        wandb_run=wandb_run,
                    )
                    logger.info(
                        f"Estimated Forward GFLOPs (2*GMACs): {results['estimated_gflops']:.4f} G"
                    )
                else:
                    logger.warning("FLOPs profiling failed.")
            except StopIteration:
                logger.warning(
                    "Could not get sample batch for FLOPs profiling (empty DataLoader?)."
                )
            except Exception as e:
                logger.error(f"FLOPs profiling failed unexpectedly: {e}", exc_info=True)

        # --- Training ---
        logger.info("Starting training phase...")
        train_config = config.get("training", {})
        algorithm_name = config.get("algorithm", {}).get("name", "").lower()
        training_fn = get_training_function(algorithm_name)

        peak_gpu_mem = float("nan")
        train_loop_start_time = time.time()
        total_energy_joules = None

        with monitor if monitor else contextlib.nullcontext() as active_monitor:
            # Different algorithms might need different arguments
            if algorithm_name == "bp":
                training_fn(
                    model,
                    train_loader,
                    val_loader,
                    config,
                    device,
                    wandb_run,
                    input_adapter,
                )
            elif algorithm_name == "cafo":
                # CaFo's train function expects model (blocks), loader, config, device, wandb
                training_fn(
                    model, train_loader, config, device, wandb_run
                )  # No adapter needed for CNN
            elif algorithm_name == "mf":
                training_fn(
                    model, train_loader, config, device, wandb_run, input_adapter
                )
            elif algorithm_name == "ff":
                training_fn(
                    model, train_loader, config, device, wandb_run, input_adapter
                )
            else:
                raise ValueError(
                    f"Unknown algorithm name for training: {algorithm_name}"
                )

        # --- Post-Training Monitoring ---
        train_loop_duration = time.time() - train_loop_start_time
        results["training_duration_sec"] = train_loop_duration
        log_metrics({"training_duration_sec": train_loop_duration}, wandb_run=wandb_run)
        logger.info(f"Training phase completed in: {format_time(train_loop_duration)}")

        if monitor:
            total_energy_joules = monitor.get_total_energy_joules()
            if total_energy_joules is not None:
                results["total_gpu_energy_joules"] = total_energy_joules
                results["total_gpu_energy_wh"] = total_energy_joules / 3600.0
                log_metrics(
                    {
                        "total_gpu_energy_joules": total_energy_joules,
                        "total_gpu_energy_wh": results["total_gpu_energy_wh"],
                    },
                    wandb_run=wandb_run,
                )
                logger.info(
                    f"Total GPU Energy Consumed (Training): {total_energy_joules:.2f} J ({results['total_gpu_energy_wh']:.4f} Wh)"
                )
            else:
                logger.warning("Energy monitoring failed to calculate total energy.")

        if nvml_active and gpu_handle:
            time.sleep(0.1)
            mem_info_end = get_gpu_memory_usage(gpu_handle)
            if mem_info_end:
                # Attempt to read peak memory usage directly if supported (newer NVML/drivers)
                try:
                    peak_mem_info = pynvml.nvmlDeviceGetMaxMemoryUsage(
                        gpu_handle
                    )  # Requires NVML 11.5+ ? Check docs
                    peak_gpu_mem = peak_mem_info / (1024**2)  # Convert bytes to MiB
                    logger.info(
                        f"Peak GPU Memory Usage (NVML Max): {peak_gpu_mem:.2f} MiB"
                    )
                except (pynvml.NVMLError_NotSupported, AttributeError):
                    logger.warning(
                        "Direct peak memory query not supported by NVML/driver. Using memory usage at end of training as proxy."
                    )
                    peak_gpu_mem = mem_info_end[
                        0
                    ]  # Use memory 'used' at the end as proxy
                except pynvml.NVMLError as e:
                    logger.error(f"NVML Error getting peak memory: {e}")
                    peak_gpu_mem = mem_info_end[0]  # Fallback to end memory

                results["peak_gpu_mem_used_mib"] = peak_gpu_mem
                log_metrics(
                    {"peak_gpu_mem_used_mib": peak_gpu_mem}, wandb_run=wandb_run
                )
                logger.info(
                    f"GPU Memory Usage (End of Training): {mem_info_end[0]:.2f} MiB Used / {mem_info_end[1]:.2f} MiB Total"
                )

        # --- Evaluation ---
        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get("evaluation", {})
        eval_criterion_name = eval_config.get("criterion", "CrossEntropyLoss")
        eval_criterion = None
        if eval_criterion_name.lower() == "crossentropyloss":
            eval_criterion = nn.CrossEntropyLoss()

        evaluation_fn = get_evaluation_function(algorithm_name)
        eval_results = {}
        test_loss_key = f"{algorithm_name.upper()}/Test_Loss"
        test_acc_key = f"{algorithm_name.upper()}/Test_Accuracy"

        # Evaluation functions might need different args
        try:
            if algorithm_name == "bp":
                eval_loss, eval_acc = evaluation_fn(
                    model, test_loader, eval_criterion, device, input_adapter
                )
                eval_results = {test_loss_key: eval_loss, test_acc_key: eval_acc}
            elif algorithm_name == "cafo":
                # CaFo eval needs predictors, assume they are attached or passed via config if needed
                cafo_eval = evaluation_fn(
                    model, test_loader, device, eval_criterion, aggregation_method="sum"
                )  # Default aggregation
                eval_results = {
                    test_loss_key: cafo_eval.get("eval_loss", float("nan")),
                    test_acc_key: cafo_eval.get("eval_accuracy", float("nan")),
                }
            elif algorithm_name == "mf":
                mf_eval = evaluation_fn(
                    model, test_loader, device, eval_criterion, input_adapter
                )
                eval_results = {
                    test_loss_key: mf_eval.get("eval_loss", float("nan")),
                    test_acc_key: mf_eval.get("eval_accuracy", float("nan")),
                }
            elif algorithm_name == "ff":
                ff_eval = evaluation_fn(model, test_loader, device, input_adapter)
                eval_results = {
                    test_acc_key: ff_eval.get("eval_accuracy", float("nan")),
                    test_loss_key: float("nan"),
                }
            else:
                raise ValueError(
                    f"Unknown algorithm name for evaluation: {algorithm_name}"
                )

        except Exception as e:
            logger.error(
                f"Evaluation failed for algorithm {algorithm_name}: {e}", exc_info=True
            )
            eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}

        logger.info(
            f"Test Set Results: Accuracy: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}"
        )
        results.update(eval_results)
        log_metrics(eval_results, wandb_run=wandb_run)

    except Exception as e:
        logger.error(f"An error occurred during the run: {e}", exc_info=True)
        results["error"] = str(e)
    finally:
        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        log_metrics({"total_run_duration_sec": total_run_time}, wandb_run=wandb_run)
        logger.info(f"Total run duration: {format_time(total_run_time)}")

        # Let atexit handle NVML shutdown
        # shutdown_nvml()

        if wandb_run and wandb_run.finish and os.environ.get("WANDB_MODE") != "offline":
            try:
                wandb_run.finish()
                logger.info("W&B run finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    return results
