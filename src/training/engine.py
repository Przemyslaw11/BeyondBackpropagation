# File: src/training/engine.py
import torch
import torch.nn as nn
import logging
import time
import os
import contextlib  # For nullcontext
from typing import Dict, Any, Optional, Tuple, Callable

# Import necessary components from our project structure
from src.utils.config_parser import (
    load_config,
)  # May not be needed here if config is passed in
from src.utils.helpers import set_seed, create_directory_if_not_exists, format_time
from src.utils.logging_utils import (
    setup_wandb,
    log_metrics,
    logger,
)  # Use the configured logger

# Import the new energy monitor class
from src.utils.monitoring import (
    init_nvml,
    shutdown_nvml,
    get_gpu_handle,
    get_gpu_memory_usage,
    GPUEnergyMonitor,
)
from src.utils.profiling import profile_model_flops
from src.data_utils.datasets import get_dataloaders

# Import base architectures
from src.architectures import FF_MLP, CaFo_CNN, MF_MLP

# Import training/evaluation functions for algorithms and baselines
from src.algorithms import train_ff_model, evaluate_ff_model  # << IMPORT FF FUNCTIONS
from src.algorithms import train_cafo_model, evaluate_cafo_model
from src.algorithms import train_mf_model, evaluate_mf_model  # Uses corrected MF
from src.baselines import train_bp_model, evaluate_bp_model


# Function get_model_and_adapter remains the same
def get_model_and_adapter(
    config: Dict[str, Any],
) -> Tuple[nn.Module, Optional[Callable]]:
    """
    Instantiates the model based on the configuration and returns an optional input adapter.
    Handles specific adaptations needed for BP baselines (e.g., adding a classifier).
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
    input_adapter = None  # Function to adapt input, e.g., flatten

    logger.info(f"Creating model architecture: {arch_name} with params: {arch_params}")
    logger.info(
        f"Input: {input_channels} channels, {image_size}x{image_size}, Num classes: {num_classes}"
    )

    if arch_name == "ff_mlp":
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        # Instantiate FF_MLP regardless, BP baseline adapts it, FF training uses it directly
        ff_model_instance = FF_MLP(**arch_params)
        if is_bp_baseline:
            logger.warning(
                "Adapting FF_MLP structure for BP baseline. Define a standard nn.Sequential MLP for ideal comparison."
            )
            layers = []
            current_dim = arch_params["input_dim"]
            for ff_layer in ff_model_instance.layers:
                layers.append(ff_layer.linear)
                layers.append(ff_layer.activation)
                current_dim = ff_layer.linear.out_features
            layers.append(nn.Linear(current_dim, num_classes))
            model = nn.Sequential(*layers)
            # BP baseline needs input flattened
            input_adapter = lambda x: x.view(x.shape[0], -1)
        else:
            # FF algorithm uses the FF_MLP instance directly
            model = ff_model_instance
            # FF algorithm's training function needs flattening handled by adapter
            input_adapter = lambda x: x.view(x.shape[0], -1)

    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
            logger.info("Creating BP baseline model from CaFo_CNN blocks.")
            with torch.no_grad():
                # Ensure dummy input matches potential device placement
                device = torch.device(config.get("general", {}).get("device", "cpu"))
                dummy_input = torch.randn(1, input_channels, image_size, image_size).to(
                    device
                )
                cafo_base.to(
                    device
                )  # Ensure base model is on device for shape calculation
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel()
            model = nn.Sequential(
                cafo_base.blocks,
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes),
            )
            cafo_base.cpu()  # Move base back if only used for shape calculation
            # No input adapter needed, CNN handles spatial input
        else:
            model = cafo_base
            # No input adapter needed for CaFo training

    elif arch_name == "mf_mlp":
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        model = MF_MLP(**arch_params)  # Use corrected MF_MLP
        # Both MF training and BP baseline need flattened input
        input_adapter = lambda x: x.view(x.shape[0], -1)

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
    Includes energy monitoring.
    """
    results = {}
    nvml_active = False
    gpu_handle = None
    monitor = None  # Initialize energy monitor variable
    run_start_time = time.time()  # Use a different name than training_start_time

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

        # Initialize NVML for monitoring (memory/energy)
        monitoring_config = config.get("monitoring", {})
        gpu_index = 0  # Default to GPU 0
        if device.type == "cuda":
            # Ensure NVML is up before getting handle or starting monitor
            if init_nvml():
                nvml_active = True
                gpu_index = device.index if device.index is not None else 0
                gpu_handle = get_gpu_handle(gpu_index)
                if gpu_handle:
                    logger.info(f"NVML active for GPU {gpu_index}.")
                    mem_info = get_gpu_memory_usage(gpu_handle)
                    if mem_info:
                        log_metrics(
                            {"initial_gpu_mem_used_mib": mem_info[0]},
                            wandb_run=wandb_run,
                        )

                    # Instantiate Energy Monitor if enabled in config
                    if monitoring_config.get("energy_enabled", True):
                        monitor = GPUEnergyMonitor(
                            device_index=gpu_index,
                            interval_sec=monitoring_config.get(
                                "energy_interval_sec", 0.2
                            ),  # Default 200ms
                        )
                        logger.info(
                            f"GPU Energy monitor initialized (Interval: {monitoring_config.get('energy_interval_sec', 0.2)}s)."
                        )
                    else:
                        logger.info("GPU Energy monitoring disabled in configuration.")
                else:
                    logger.warning(
                        f"NVML active but failed to get handle for GPU {gpu_index}. Monitoring disabled."
                    )
                    nvml_active = False  # Treat as inactive if handle fails
            else:
                logger.warning("NVML initialization failed. GPU monitoring disabled.")
        else:
            logger.info("Running on CPU, GPU monitoring disabled.")

        # --- Data Loading ---
        data_config = config.get("dataset", {})
        loader_config = config.get("data_loader", {})
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root_dir", "./data"),
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
        model, input_adapter = get_model_and_adapter(config)
        model.to(device)
        logger.info(
            f"Model '{config.get('model', {}).get('name')}' instantiated and moved to {device}."
        )
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

        # --- FLOPs Profiling ---
        profiling_config = config.get("profiling", {})
        if profiling_config.get("enabled", True):
            try:
                sample_input, _ = next(iter(train_loader))
                # Apply adapter if needed for profiling (use shape of *adapted* input)
                profile_input_sample = (
                    input_adapter(sample_input) if input_adapter else sample_input
                )
                input_shape = profile_input_sample.shape
                logger.info(f"Profiling FLOPs with input shape: {input_shape}")
                # Profile the model currently configured (could be BP baseline version)
                gmacs = profile_model_flops(
                    model,
                    input_shape=input_shape,
                    device=device,
                    verbose=profiling_config.get("verbose", False),
                )
                if gmacs is not None:
                    results["estimated_gmacs"] = gmacs
                    results["estimated_gflops"] = gmacs * 2.0
                    log_metrics(
                        {"estimated_gmacs": gmacs, "estimated_gflops": gmacs * 2.0},
                        wandb_run=wandb_run,
                    )
                    logger.info(
                        f"Estimated GFLOPs: {results['estimated_gflops']:.4f} G"
                    )
                else:
                    logger.warning("FLOPs profiling failed.")
            except Exception as e:
                logger.error(f"FLOPs profiling failed: {e}", exc_info=True)

        # --- Training ---
        logger.info("Starting training phase...")
        train_config = config.get("training", {})
        # Use a distinct key for algorithm-specific training params if needed
        algo_train_config = config.get(
            "algorithm_params", train_config
        )  # Fallback to main training config

        algorithm_name = config.get("algorithm", {}).get("name", "").lower()
        peak_gpu_mem = 0.0
        train_loop_start_time = time.time()
        total_energy_joules = None

        # Use the energy monitor context manager around the training call
        try:
            with monitor if monitor else contextlib.nullcontext() as active_monitor:
                # Core training logic based on algorithm
                if algorithm_name == "bp":
                    # BP uses the main training config (epochs, optimizer settings from tuning etc.)
                    train_bp_model(
                        model,
                        train_loader,
                        val_loader,
                        train_config,
                        device,
                        wandb_run,
                        input_adapter,
                    )
                elif algorithm_name == "cafo":
                    # CaFo uses its specific params (epochs_per_block etc.)
                    train_cafo_model(
                        model, train_loader, algo_train_config, device, wandb_run
                    )  # Pass algo config
                elif algorithm_name == "mf":
                    # MF uses its specific params (epochs_per_layer etc.)
                    train_mf_model(
                        model,
                        train_loader,
                        algo_train_config,
                        device,
                        wandb_run,
                        input_adapter,
                    )  # Pass algo config
                elif algorithm_name == "ff":
                    # FF uses its specific params (epochs_per_layer etc.)
                    train_ff_model(
                        model,
                        train_loader,
                        algo_train_config,
                        device,
                        wandb_run,
                        input_adapter,
                    )  # Pass algo config
                else:
                    raise ValueError(
                        f"Unknown algorithm name for training: {algorithm_name}"
                    )

        finally:  # Ensure duration calculation and energy retrieval happens
            train_loop_duration = time.time() - train_loop_start_time
            results["training_duration_sec"] = train_loop_duration
            log_metrics(
                {"training_duration_sec": train_loop_duration}, wandb_run=wandb_run
            )
            logger.info(
                f"Training phase completed in: {format_time(train_loop_duration)}"
            )

            # Retrieve energy if monitor was used
            if monitor:
                total_energy_joules = monitor.get_total_energy()
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
                        f"Total GPU Energy Consumed (Training Loop): {total_energy_joules:.2f} J ({results['total_gpu_energy_wh']:.4f} Wh)"
                    )
                else:
                    logger.warning(
                        "Energy monitoring ran, but failed to calculate total energy (likely too few samples)."
                    )

            # Log peak memory after training
            if nvml_active and gpu_handle:
                time.sleep(0.1)  # Allow time for memory readings to stabilize
                mem_info = get_gpu_memory_usage(gpu_handle)
                if mem_info:
                    peak_gpu_mem = mem_info[0]
                    results["peak_gpu_mem_used_mib"] = peak_gpu_mem
                    log_metrics(
                        {"peak_gpu_mem_used_mib": peak_gpu_mem}, wandb_run=wandb_run
                    )
                    logger.info(
                        f"GPU Memory Usage (End of Training): {peak_gpu_mem:.2f} MiB Used / {mem_info[1]:.2f} MiB Total"
                    )

        # --- Evaluation ---
        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get("evaluation", {})
        eval_criterion_name = eval_config.get("criterion", "CrossEntropyLoss")
        eval_criterion = None
        if eval_criterion_name.lower() == "crossentropyloss":
            eval_criterion = nn.CrossEntropyLoss()
        else:
            logger.warning(
                f"Using evaluation criterion: {eval_criterion_name} or None if not CE."
            )
            # Handle other criteria if necessary or allow None

        eval_results = {}
        test_loss_key = "test_loss"
        test_acc_key = "test_accuracy"

        if algorithm_name == "bp":
            eval_loss, eval_acc = evaluate_bp_model(
                model, test_loader, eval_criterion, device, input_adapter
            )
            eval_results = {test_loss_key: eval_loss, test_acc_key: eval_acc}
        elif algorithm_name == "cafo":
            # Default to last predictor for final eval, use consistent key names
            cafo_predictor_idx = -1  # Or make configurable in eval_config
            cafo_eval = evaluate_cafo_model(
                model,
                test_loader,
                device,
                eval_criterion,
                use_predictor_index=cafo_predictor_idx,
            )
            last_predictor_idx_eval = len(
                model.predictors
            )  # Index used in keys is 1-based
            eval_results = {
                test_loss_key: cafo_eval.get(
                    f"eval_predictor_{last_predictor_idx_eval}_loss", float("nan")
                ),
                test_acc_key: cafo_eval.get(
                    f"eval_predictor_{last_predictor_idx_eval}_accuracy", float("nan")
                ),
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
            ff_eval_results = evaluate_ff_model(
                model, test_loader, device, input_adapter
            )
            eval_results = {
                test_acc_key: ff_eval_results.get("eval_accuracy", float("nan")),
                # FF doesn't typically calculate loss in this way
                test_loss_key: float("nan"),
            }
        else:
            raise ValueError(f"Unknown algorithm name for evaluation: {algorithm_name}")

        logger.info(
            f"Test Set Evaluation: Accuracy: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}"
        )
        # Update results dict, using consistent keys
        results[test_acc_key] = eval_results.get(test_acc_key)
        results[test_loss_key] = eval_results.get(test_loss_key)
        # Log final test metrics to wandb
        log_metrics(
            {
                test_loss_key: results[test_loss_key],
                test_acc_key: results[test_acc_key],
            },
            wandb_run=wandb_run,
        )

    except Exception as e:
        logger.error(f"An error occurred during the run: {e}", exc_info=True)
        results["error"] = str(e)
        # raise e # Optional: re-raise
    finally:
        # --- Cleanup ---
        # NVML shutdown handled by atexit hook

        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        log_metrics({"total_run_duration_sec": total_run_time}, wandb_run=wandb_run)
        logger.info(f"Total run duration: {format_time(total_run_time)}")

    return results


# --- Keep __main__ block for testing engine ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Testing Training Engine (with FF/Energy)...")

    # Example: Modify base config content for testing FF
    base_config_ff_test = """
general:
  seed: 1234
  device: cuda # Force CUDA if available for energy/FF test
dataset:
  name: FashionMNIST
  root_dir: ./data_test_engine_ff
  num_classes: 10
  input_channels: 1
  image_size: 28
data_loader:
  batch_size: 8
  num_workers: 0
monitoring:
  enabled: true
  energy_enabled: true
  energy_interval_sec: 0.2
profiling:
  enabled: false
training: # General training section (may not be used by FF)
  epochs: 1
evaluation:
  criterion: CrossEntropyLoss # Not really used by FF eval
"""
    ff_config_content = """
algorithm: {name: FF}
model:
  name: FF_MLP
  params:
    # input_dim: 784 # Calculated in get_model
    hidden_dims: [64] # Very small FF MLP
    activation: relu
    # num_classes: 10 # Calculated in get_model
algorithm_params: # Params specifically for train_ff_model
  optimizer_type: Adam
  lr: 0.01
  weight_decay: 0.0
  threshold: 1.5
  epochs_per_layer: 1 # Train each layer for 1 epoch
  log_interval: 1
"""
    os.makedirs("configs_test_engine", exist_ok=True)
    with open("configs_test_engine/base_test_ff.yaml", "w") as f:
        f.write(base_config_ff_test)
    with open("configs_test_engine/ff_test.yaml", "w") as f:
        f.write(ff_config_content)

    print("\n--- Testing FF Run ---")
    if torch.cuda.is_available():
        try:
            config_ff_run = load_config(
                "configs_test_engine/ff_test.yaml",
                base_config_path="configs_test_engine/base_test_ff.yaml",
            )
            results_ff = run_training(config_ff_run, wandb_run=None)
            print("FF Run Results:", results_ff)
            assert "error" not in results_ff
            assert "test_accuracy" in results_ff
            if config_ff_run.get("monitoring", {}).get("energy_enabled", True):
                assert "total_gpu_energy_joules" in results_ff
                assert (
                    results_ff["total_gpu_energy_joules"] >= 0
                )  # Allow 0 if very fast
        except Exception as e:
            print(f"FF run failed: {e}", exc_info=True)
    else:
        print("Skipping FF test: CUDA not available.")

    # Consider adding tests for corrected MF and CaFo here as well if needed

    # Clean up dummy configs and data
    import shutil

    # shutil.rmtree('configs_test_engine')
    # if os.path.exists('./data_test_engine_ff'): shutil.rmtree('./data_test_engine_ff')
    print(
        "\nEngine testing finished. Manual cleanup of 'configs_test_engine' and data dirs might be needed."
    )
