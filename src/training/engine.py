# File: src/training/engine.py
import torch
import torch.nn as nn
import logging
import time
import os
import contextlib
from typing import Dict, Any, Optional, Tuple, Callable

# Import project components
from src.utils.config_parser import load_config
from src.utils.helpers import set_seed, create_directory_if_not_exists, format_time
from src.utils.logging_utils import (
    setup_wandb,
    log_metrics,
    logger,
    setup_logging,
)  # Import setup_logging
from src.utils.monitoring import (
    init_nvml,
    shutdown_nvml,
    get_gpu_handle,
    get_gpu_memory_usage,
    GPUEnergyMonitor,
)
from src.utils.profiling import profile_model_flops
from src.data_utils.datasets import get_dataloaders

# Architectures
from src.architectures import FF_MLP, CaFo_CNN, MF_MLP

# Algorithms & Baselines
from src.algorithms import train_ff_model, evaluate_ff_model
from src.algorithms import train_cafo_model, evaluate_cafo_model
from src.algorithms import train_mf_model, evaluate_mf_model
from src.baselines import train_bp_model, evaluate_bp_model


def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device  # Pass device for shape calculation
) -> Tuple[nn.Module, Optional[Callable]]:
    """
    Instantiates the model based on the configuration and returns an optional input adapter.
    Handles specific adaptations needed for BP baselines.
    """
    model_config = config.get("model", {})
    arch_name = model_config.get("name", "").lower()
    arch_params = model_config.get("params", {})
    dataset_config = config.get("dataset", {})
    num_classes = dataset_config.get("num_classes", 10)
    input_channels = dataset_config.get("input_channels", 1)
    image_size = dataset_config.get("image_size", 28)

    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    model = None
    input_adapter = None

    logger.info(
        f"Creating model architecture: {arch_name} (BP Baseline: {is_bp_baseline})"
    )
    # logger.info(f"Architecture params: {arch_params}")
    # logger.info(f"Dataset params: Channels={input_channels}, Size={image_size}, Classes={num_classes}")

    # --- Determine necessary parameters if not provided ---
    if arch_name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
            logger.debug(f"Calculated input_dim: {arch_params['input_dim']}")
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes

    # --- Instantiate based on architecture name ---
    if arch_name == "ff_mlp":
        ff_model_instance = FF_MLP(**arch_params)
        if is_bp_baseline:
            # Create a standard MLP from FF_Layer parameters for the BP baseline
            logger.info(
                "Adapting FF_MLP structure for BP baseline -> Standard nn.Sequential MLP."
            )
            layers = []
            # Input adapter needed for BP MLP
            input_adapter = lambda x: x.view(x.shape[0], -1)
            current_dim = arch_params["input_dim"]  # Use original image dim
            for (
                ff_layer
            ) in ff_model_instance.layers:  # Iterate through FF_Layer modules
                layers.append(
                    nn.Linear(
                        current_dim,
                        ff_layer.linear.out_features,
                        bias=ff_layer.linear.bias is not None,
                    )
                )
                # Copy weights if needed, or reinitialize for BP tuning
                # layers[-1].load_state_dict(ff_layer.linear.state_dict()) # Optional: Start BP from FF weights
                layers.append(type(ff_layer.activation)())  # Instantiate new activation
                current_dim = ff_layer.linear.out_features
            layers.append(
                nn.Linear(current_dim, num_classes)
            )  # Add final BP classifier
            model = nn.Sequential(*layers)
        else:
            # FF algorithm uses the FF_MLP instance directly
            model = ff_model_instance
            # FF training needs flattening adapter before label embedding
            input_adapter = lambda x: x.view(x.shape[0], -1)

    elif arch_name == "cafo_cnn":
        # Instantiate the base CaFo CNN (contains only blocks now)
        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
            logger.info(
                "Creating BP baseline model from CaFo_CNN blocks + final Linear layer."
            )
            # Calculate output features from the last block
            with torch.no_grad():
                cafo_base.to(device)  # Move to device for calculation
                dummy_input = torch.randn(1, input_channels, image_size, image_size).to(
                    device
                )
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel()
                cafo_base.cpu()  # Move back if only used for shape calculation
                logger.debug(
                    f"Flattened output dimension from CaFo blocks: {num_output_features}"
                )

            model = nn.Sequential(
                cafo_base.blocks,  # Use the ModuleList of blocks
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes),
            )
            # No input adapter usually needed for CNN baselines
        else:
            # CaFo algorithm uses the CaFo_CNN instance containing blocks directly
            # Predictors are handled by train_cafo_model
            model = cafo_base
            # No input adapter needed

    elif arch_name == "mf_mlp":
        # Instantiate MF_MLP (includes W layers and M parameters)
        model = MF_MLP(**arch_params)
        # Both MF training and BP baseline need flattened input
        input_adapter = lambda x: x.view(x.shape[0], -1)
        if is_bp_baseline:
            logger.info("Using standard forward pass of MF_MLP for BP baseline.")
            # BP training will ignore the projection matrices automatically

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

    # Ensure logging is configured (can be called multiple times safely if needed)
    # setup_logging(log_level=config.get('logging', {}).get('level', 'INFO')) # Setup happens in run_experiment.py

    try:
        # --- Setup ---
        general_config = config.get("general", {})
        seed = general_config.get("seed", 42)
        set_seed(seed)
        logger.info(f"Using random seed: {seed}")
        device_name = general_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(device_name)
        logger.info(f"Using device: {device}")

        # --- W&B Initialization ---
        # Moved W&B setup here to capture the whole run
        if (
            wandb_run is None
        ):  # Allow passing an existing run (e.g., from Optuna wrapper)
            wandb_run = setup_wandb(config)  # Use config for project/entity/name

        # --- Monitoring Initialization ---
        monitoring_config = config.get("monitoring", {})
        if device.type == "cuda":
            if init_nvml():
                nvml_active = True
                gpu_index = device.index if device.index is not None else 0
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
        data_config = config.get("data", {})  # Use 'data' section now
        loader_config = config.get("data_loader", {})
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),  # Use 'root' key
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
        model, input_adapter = get_model_and_adapter(config, device)  # Pass device
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
                # Get a sample input and apply adapter to determine profiling shape/input
                sample_input_img, _ = next(iter(train_loader))
                if input_adapter:
                    profile_input_constructor = lambda: input_adapter(sample_input_img)
                else:
                    profile_input_constructor = lambda: sample_input_img

                logger.info(f"Profiling FLOPs...")
                gmacs = profile_model_flops(
                    model,
                    input_constructor=profile_input_constructor,
                    device=device,
                    verbose=profiling_config.get("verbose", False),
                )
                if gmacs is not None:
                    results["estimated_gmacs"] = gmacs
                    results["estimated_gflops"] = gmacs * 2.0  # Standard estimation
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
        algo_train_config = config.get(
            "algorithm_params", train_config
        )  # Use specific section first
        algorithm_name = config.get("algorithm", {}).get("name", "").lower()

        peak_gpu_mem = float("nan")  # Default if not measurable
        train_loop_start_time = time.time()
        total_energy_joules = None

        with monitor if monitor else contextlib.nullcontext() as active_monitor:
            if algorithm_name == "bp":
                train_bp_model(
                    model,
                    train_loader,
                    val_loader,
                    config,
                    device,
                    wandb_run,
                    input_adapter,
                )  # Pass full config for now
            elif algorithm_name == "cafo":
                train_cafo_model(
                    model, train_loader, config, device, wandb_run
                )  # Pass full config
            elif algorithm_name == "mf":
                train_mf_model(
                    model, train_loader, config, device, wandb_run, input_adapter
                )  # Pass full config
            elif algorithm_name == "ff":
                train_ff_model(
                    model, train_loader, config, device, wandb_run, input_adapter
                )  # Pass full config
            else:
                raise ValueError(
                    f"Unknown algorithm name for training: {algorithm_name}"
                )

        # --- Post-Training Monitoring ---
        train_loop_duration = time.time() - train_loop_start_time
        results["training_duration_sec"] = train_loop_duration
        log_metrics({"training_duration_sec": train_loop_duration}, wandb_run=wandb_run)
        logger.info(f"Training phase completed in: {format_time(train_loop_duration)}")

        if monitor:  # Retrieve energy from monitor if it exists
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

        if nvml_active and gpu_handle:  # Log peak memory
            time.sleep(0.1)  # Short pause
            mem_info_end = get_gpu_memory_usage(gpu_handle)
            if mem_info_end:
                peak_gpu_mem = mem_info_end[
                    0
                ]  # Use memory 'used' at the end as proxy for peak
                results["peak_gpu_mem_used_mib"] = peak_gpu_mem
                log_metrics(
                    {"peak_gpu_mem_used_mib": peak_gpu_mem}, wandb_run=wandb_run
                )
                logger.info(
                    f"GPU Memory Usage (End of Training): {peak_gpu_mem:.2f} MiB Used / {mem_info_end[1]:.2f} MiB Total"
                )

        # --- Evaluation ---
        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get("evaluation", {})
        eval_criterion_name = eval_config.get("criterion", "CrossEntropyLoss")
        eval_criterion = None
        if eval_criterion_name.lower() == "crossentropyloss":
            eval_criterion = nn.CrossEntropyLoss()

        eval_results = {}
        test_loss_key = "test_loss"
        test_acc_key = "test_accuracy"

        if algorithm_name == "bp":
            eval_loss, eval_acc = evaluate_bp_model(
                model, test_loader, eval_criterion, device, input_adapter
            )
            eval_results = {test_loss_key: eval_loss, test_acc_key: eval_acc}
        elif algorithm_name == "cafo":
            # Evaluate using sum aggregation by default
            cafo_eval = evaluate_cafo_model(
                model, test_loader, device, eval_criterion, aggregation_method="sum"
            )
            eval_results = {
                test_loss_key: cafo_eval.get("eval_loss", float("nan")),
                test_acc_key: cafo_eval.get("eval_accuracy", float("nan")),
            }
        elif algorithm_name == "mf":
            mf_eval = evaluate_mf_model(
                model, test_loader, device, eval_criterion, input_adapter
            )
            eval_results = {
                test_loss_key: mf_eval.get("eval_loss", float("nan")),
                test_acc_key: mf_eval.get("eval_accuracy", float("nan")),
            }
        elif algorithm_name == "ff":
            ff_eval = evaluate_ff_model(model, test_loader, device, input_adapter)
            eval_results = {
                test_acc_key: ff_eval.get("eval_accuracy", float("nan")),
                test_loss_key: float("nan"),  # FF eval doesn't calculate loss this way
            }
        else:
            raise ValueError(f"Unknown algorithm name for evaluation: {algorithm_name}")

        logger.info(
            f"Test Set Results: Accuracy: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}"
        )
        results.update(eval_results)  # Add eval results to main results dict
        log_metrics(eval_results, wandb_run=wandb_run)  # Log to W&B

    except Exception as e:
        logger.error(f"An error occurred during the run: {e}", exc_info=True)
        results["error"] = str(e)
        # raise e # Optional: re-raise to halt execution
    finally:
        # --- Cleanup ---
        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        log_metrics({"total_run_duration_sec": total_run_time}, wandb_run=wandb_run)
        logger.info(f"Total run duration: {format_time(total_run_time)}")

        # Shutdown NVML explicitly (though atexit should also handle it)
        # shutdown_nvml() # Let atexit handle it to avoid issues if multiple runs happen

        # Finish W&B run if it was initialized here
        if wandb_run and wandb_run.finish:
            try:
                wandb_run.finish()
                logger.info("W&B run finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    return results


# Removed the __main__ block for cleaner engine file
