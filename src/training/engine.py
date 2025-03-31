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

# ... (get_model_and_adapter function remains unchanged) ...


def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device
) -> Tuple[
    nn.Module, Optional[Callable[[torch.Tensor], torch.Tensor]]
]:  # Added Tensor type hints
    """
    Instantiates the model based on the configuration and returns an optional input adapter.
    Handles specific adaptations needed for BP baselines.

    Returns:
        A tuple containing:
            - The instantiated PyTorch model (nn.Module).
            - An optional callable function (input_adapter) that preprocesses the input batch
              (e.g., flattens it) before it's fed to the model's forward method.
              Returns None if no adaptation is needed.
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

    model: Optional[nn.Module] = None
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = (
        None  # Defined here
    )

    logger.info(
        f"Creating model architecture: {arch_name} (BP Baseline: {is_bp_baseline})"
    )

    # --- Determine necessary parameters and adapter based on architecture ---
    if arch_name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        # MLPs require flattened input
        input_adapter = lambda x: x.view(x.shape[0], -1)
        logger.debug(f"Architecture {arch_name} requires input flattening adapter.")
    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        # CNNs usually don't need a specific adapter from this function
        input_adapter = None
    else:
        # Raise error for unknown architectures earlier
        raise ValueError(f"Unknown model architecture name in config: {arch_name}")

    # --- Instantiate based on architecture name ---
    if arch_name == "ff_mlp":
        # Instantiate the base FF_MLP regardless
        ff_model_instance = FF_MLP(**arch_params)
        if is_bp_baseline:
            logger.info(
                "Adapting FF_MLP structure for BP baseline -> Standard nn.Sequential MLP."
            )
            layers = []
            # BP baseline uses original image dim before flattening by adapter
            bp_input_dim = arch_params["input_dim"]
            # Reconstruct sequential layers from FF_MLP structure
            # Input Adapter Layer of FF becomes first Linear layer for BP
            layers.append(
                nn.Linear(
                    bp_input_dim,  # Use flattened input dim
                    ff_model_instance.input_adapter_layer.out_features,  # Corrected reference
                    bias=ff_model_instance.input_adapter_layer.bias
                    is not None,  # Corrected reference
                )
            )
            layers.append(type(ff_model_instance.first_layer_activation)())
            current_dim = (
                ff_model_instance.input_adapter_layer.out_features
            )  # Corrected reference
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
            # Note: FF's internal training logic handles its specific label embedding,
            # the 'input_adapter' here is just for the image flattening part if needed.

    elif arch_name == "cafo_cnn":
        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
            logger.info(
                "Creating BP baseline model from CaFo_CNN blocks + final Linear layer."
            )
            # Temporarily move model to device to calculate output dim
            cafo_base.to(device)
            with torch.no_grad():
                # Use dummy input dimensions directly from config
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
                cafo_base.blocks,  # Blocks first
                nn.Flatten(),  # Then flatten
                nn.Linear(num_output_features, num_classes),  # Then linear
            )
        else:
            model = cafo_base  # CaFo training uses the base model

    elif arch_name == "mf_mlp":
        # MF_MLP instance contains W layers, M parameters, and standard forward for BP
        model = MF_MLP(**arch_params)
        if is_bp_baseline:
            logger.info("Using standard forward pass of MF_MLP for BP baseline.")
        # MF training function uses specific forward methods and the adapter

    # Model should be instantiated by now if arch_name was valid
    if model is None:
        # This condition should ideally not be reached due to the check at the start
        raise RuntimeError(f"Failed to instantiate model for architecture: {arch_name}")

    logger.info(
        f"Model '{arch_name}' and input adapter (type: {type(input_adapter)}) created."
    )
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

        # --- Model Instantiation (Get model AND adapter) ---
        model, input_adapter = get_model_and_adapter(config, device)  # Get both here
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

        # --- FLOPs Profiling (Pass adapter if needed) ---
        profiling_config = config.get("profiling", {})
        if profiling_config.get("enabled", True):
            try:
                sample_input_img, _ = next(iter(train_loader))
                sample_input_device = sample_input_img.to(device)

                # Create the constructor using the adapter if it exists
                if input_adapter:
                    # The constructor now applies the adapter before profiling
                    profile_input_constructor = lambda: input_adapter(
                        sample_input_device
                    )
                    logger.debug("Using input adapter for FLOPs profiling.")
                else:
                    profile_input_constructor = lambda: sample_input_device
                    logger.debug("No input adapter used for FLOPs profiling.")

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

        # --- Training (Pass adapter consistently) ---
        logger.info("Starting training phase...")
        train_config = config.get("training", {})
        algorithm_name = config.get("algorithm", {}).get("name", "").lower()
        training_fn = get_training_function(algorithm_name)

        peak_gpu_mem = float("nan")
        train_loop_start_time = time.time()
        total_energy_joules = None

        with monitor if monitor else contextlib.nullcontext() as active_monitor:
            # Determine arguments needed by the specific training function
            training_args = {
                "model": model,
                "train_loader": train_loader,
                "config": config,
                "device": device,
                "wandb_run": wandb_run,
            }
            if algorithm_name == "bp":
                training_args["val_loader"] = val_loader
                training_args["input_adapter"] = input_adapter
            elif algorithm_name in ["mf"]:  # FF no longer needs it here
                training_args["input_adapter"] = input_adapter
            # CaFo doesn't use a generic input adapter here; handled internally
            # FF no longer needs input_adapter passed to train_ff_model

            # Call the training function with appropriate arguments
            training_fn(**training_args)

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
            # Short delay might help capture peak memory after training finishes
            time.sleep(0.1)
            try:
                # Use torch cuda memory stats as primary source for peak memory if available
                # Reset peak stats at the beginning of training for accurate measurement
                # This might need to be done *before* the training loop starts
                # torch.cuda.reset_peak_memory_stats(device) # Call this before training loop
                # peak_bytes = torch.cuda.max_memory_allocated(device)
                # peak_gpu_mem = peak_bytes / (1024**2) # Convert bytes to MiB
                # logger.info(f"Peak GPU Memory Usage (torch.cuda): {peak_gpu_mem:.2f} MiB")

                # Fallback/Cross-check with NVML
                mem_info_end = get_gpu_memory_usage(gpu_handle)
                if mem_info_end:
                    try:
                        # Newer NVML check
                        peak_mem_info = pynvml.nvmlDeviceGetMaxMemoryUsage(gpu_handle)
                        peak_gpu_mem_nvml = peak_mem_info / (1024**2)
                        logger.info(
                            f"Peak GPU Memory Usage (NVML Max): {peak_gpu_mem_nvml:.2f} MiB"
                        )
                        peak_gpu_mem = (
                            peak_gpu_mem_nvml  # Prefer NVML direct query if available
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
                        peak_gpu_mem = mem_info_end[0]  # Fallback

                    results["peak_gpu_mem_used_mib"] = peak_gpu_mem
                    log_metrics(
                        {"peak_gpu_mem_used_mib": peak_gpu_mem}, wandb_run=wandb_run
                    )
                    logger.info(
                        f"GPU Memory Usage (End of Training): {mem_info_end[0]:.2f} MiB Used / {mem_info_end[1]:.2f} MiB Total"
                    )
            except Exception as e_mem:
                logger.error(f"Failed to get memory usage: {e_mem}", exc_info=True)
                results["peak_gpu_mem_used_mib"] = float("nan")

        # --- Evaluation (Pass adapter consistently) ---
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

        # Determine arguments needed by the specific evaluation function
        eval_args = {
            "model": model,
            "data_loader": test_loader,
            "device": device,
        }
        if algorithm_name in [
            "bp",
            "mf",
            "cafo",
        ]:  # CaFo eval might use criterion internally
            eval_args["criterion"] = eval_criterion
        if algorithm_name in [
            "bp",
            "mf",
            "ff",
        ]:  # FF eval also uses input adapter (handled inside evaluate_ff_model now)
            eval_args["input_adapter"] = input_adapter
        if algorithm_name == "cafo":
            eval_args["predictors"] = getattr(model, "trained_predictors", None)
            eval_args["aggregation_method"] = config.get("algorithm_params", {}).get(
                "aggregation_method", "sum"
            )

        try:
            eval_output = evaluation_fn(**eval_args)

            # Process output based on what eval function returns
            if isinstance(eval_output, dict):  # Preferred return type
                eval_results = {
                    test_loss_key: eval_output.get("eval_loss", float("nan")),
                    test_acc_key: eval_output.get("eval_accuracy", float("nan")),
                }
            elif (
                isinstance(eval_output, tuple) and len(eval_output) == 2
            ):  # e.g., BP baseline return
                eval_results = {
                    test_loss_key: eval_output[0],
                    test_acc_key: eval_output[1],
                }
            else:
                logger.error(
                    f"Unexpected return type from evaluation function: {type(eval_output)}"
                )
                eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}

        except Exception as e:
            logger.error(
                f"Evaluation failed for algorithm {algorithm_name}: {e}", exc_info=True
            )
            eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}

        logger.info(
            f"Test Set Results: Accuracy: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}"
        )
        results.update(eval_results)
        # Ensure keys logged to W&B are sanitized if necessary (e.g., no slashes if not grouped)
        log_metrics(
            {
                "Test/Loss": eval_results.get(test_loss_key, float("nan")),
                "Test/Accuracy": eval_results.get(test_acc_key, float("nan")),
            },
            wandb_run=wandb_run,
        )

    except Exception as e:
        logger.critical(
            f"A critical error occurred during the run: {e}", exc_info=True
        )  # Use critical for top-level crash
        results["error"] = str(e)
        # Ensure W&B run is finished even on error if initialized
        if wandb_run and wandb_run.finish:
            try:
                wandb_run.finish(exit_code=1)  # Indicate error exit
                logger.info("W&B run finished with error code.")
            except Exception as e_wandb:
                logger.error(f"Error finishing W&B run after exception: {e_wandb}")
        # Re-raise the exception after logging and cleanup attempt
        raise e
    finally:
        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        log_metrics({"total_run_duration_sec": total_run_time}, wandb_run=wandb_run)
        logger.info(f"Total run duration: {format_time(total_run_time)}")

        # Let atexit handle NVML shutdown

        # Finish W&B run if it hasn't been finished due to error
        if (
            wandb_run
            and wandb_run.finish
            and results.get("error") is None
            and os.environ.get("WANDB_MODE") != "offline"
        ):
            try:
                wandb_run.finish()
                logger.info("W&B run finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    return results
