# File: src/training/engine.py
import torch
import torch.nn as nn
import logging
import time
import os
import contextlib
import pynvml
from typing import Dict, Any, Optional, Tuple, Callable, List

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
# Import the modified FF_MLP
from src.architectures import FF_MLP, CaFo_CNN, MF_MLP
from src.algorithms import (
    get_training_function,
    get_evaluation_function,
)

# --- get_model_and_adapter function (MODIFIED for updated FF_MLP) ---
def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device
) -> Tuple[ nn.Module, Optional[Callable[[torch.Tensor], torch.Tensor]] ]:
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
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    logger.info(f"Creating model arch: {arch_name} (Algo: {algorithm_name.upper()})")

    valid_baseline_architectures = ["ff_mlp", "cafo_cnn", "mf_mlp"] # ff_mlp now refers to the modified one
    if is_bp_baseline and arch_name not in valid_baseline_architectures:
        logger.warning(f"BP baseline uses arch '{arch_name}', check correspondence.")

    # --- Determine params and adapter ---
    if arch_name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params: arch_params["input_dim"] = input_channels * image_size * image_size
        if "num_classes" not in arch_params: arch_params["num_classes"] = num_classes
        input_adapter = lambda x: x.view(x.shape[0], -1) # Flatten input
        logger.debug(f"Arch {arch_name} requires input flattening adapter.")
    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params: arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params: arch_params["image_size"] = image_size
        if "num_classes" not in arch_params: arch_params["num_classes"] = num_classes
        input_adapter = None
        logger.debug(f"Arch {arch_name} does not require standard input adapter.")
    else: raise ValueError(f"Unknown model architecture: {arch_name}")

    # --- Instantiate ---
    if arch_name == "ff_mlp":
        if is_bp_baseline:
            logger.info("Adapting modified FF_MLP structure for BP baseline -> Standard nn.Sequential MLP.")
            bp_input_dim = arch_params["input_dim"]
            hidden_dims = arch_params.get("hidden_dims", [])
            activation_name = arch_params.get("activation", "ReLU").lower()
            use_bias = arch_params.get("bias", True)
            if not hidden_dims: raise ValueError("BP baseline creation failed: hidden_dims missing for FF_MLP.")
            layers = []
            current_dim = bp_input_dim
            act_cls = nn.ReLU if activation_name == 'relu' else nn.Tanh # Use standard activation
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
                layers.append(act_cls())
                current_dim = h_dim
            # Add final layer for classification
            layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
            model = nn.Sequential(*layers)
            logger.debug("Created BP baseline Sequential model from modified FF_MLP spec.")
        else:
            # Instantiate the modified FF_MLP, passing full config and device
            # Ensure arch_params are passed too, as __init__ uses them
            model = FF_MLP(config=config, device=device, **arch_params)
            logger.debug("Using native modified FF_MLP structure.")

    elif arch_name == "cafo_cnn":
        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
            logger.info("Creating BP baseline model from CaFo_CNN blocks + final Linear layer.")
            cafo_base.to(device)
            with torch.no_grad():
                dummy_input = torch.randn(1, input_channels, image_size, image_size).to(device)
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel()
            cafo_base.cpu()
            logger.debug(f"Flattened output dimension from CaFo blocks: {num_output_features}")
            model = nn.Sequential(cafo_base.blocks, nn.Flatten(), nn.Linear(num_output_features, num_classes))
            logger.debug("Created BP baseline Sequential model from CaFo_CNN spec.")
        else:
            model = cafo_base
            logger.debug("Using native CaFo_CNN structure.")

    elif arch_name == "mf_mlp":
        model = MF_MLP(**arch_params)
        if is_bp_baseline: logger.info("Using standard forward pass of MF_MLP for BP baseline.")
        else: logger.debug("Using native MF_MLP structure.")

    if model is None: raise RuntimeError(f"Failed to instantiate model: {arch_name}")
    logger.info(f"Model '{arch_name}' (Algo: {algorithm_name.upper()}) created.")
    return model, input_adapter


# --- run_training (Ensure val_loader passed to FF) ---
def run_training( config: Dict[str, Any], wandb_run: Optional[Any] = None ) -> Dict[str, Any]:
    results = {}
    nvml_active = False
    gpu_handle = None
    monitor = None
    run_start_time = time.time()
    step_ref = [-1]
    peak_gpu_mem_during_training = float('nan')
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()

    try:
        # Setup (Seed, Device, W&B, Monitoring)
        general_config = config.get("general", {})
        seed = general_config.get("seed", 42)
        set_seed(seed)
        logger.info(f"Using random seed: {seed}")

        device_preference = general_config.get("device", "auto").lower()
        if device_preference == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_preference == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} (Preference: '{device_preference}')")

        if wandb_run is None:
            wandb_run = setup_wandb(config, job_type="training")
        if wandb_run:
            try:
                wandb_run.define_metric("global_step", summary="max")
                wandb_run.define_metric("*", step_metric="global_step")
                logger.info("Defined 'global_step' as the default x-axis metric for W&B.")
            except Exception as e_define:
                logger.error(f"Failed to define W&B metric 'global_step': {e_define}")

        monitoring_config = config.get("monitoring", {})
        initial_metrics = {}
        if device.type == "cuda":
            if init_nvml():
                nvml_active = True
                gpu_index = torch.cuda.current_device()
                gpu_handle = get_gpu_handle(gpu_index)
                if gpu_handle:
                    logger.info(f"NVML active for GPU {gpu_index}.")
                    mem_info_start = get_gpu_memory_usage(gpu_handle)
                    if mem_info_start:
                        initial_metrics["initial_gpu_mem_used_mib"] = mem_info_start[0]
                else:
                    logger.warning(f"NVML active but failed get handle for GPU {gpu_index}.")
                    nvml_active = False # Treat as inactive if handle fails

                # Initialize monitor only if handle is valid and energy monitoring is enabled
                if monitoring_config.get("energy_enabled", True) and gpu_handle:
                    monitor = GPUEnergyMonitor(device_index=gpu_index, interval_sec=monitoring_config.get("energy_interval_sec", 0.2))
                    logger.info(f"GPU Energy monitor initialized (Interval: {monitor._interval_sec}s).")
                elif monitoring_config.get("energy_enabled", True):
                    logger.warning("Energy monitoring enabled but failed to get GPU handle.")
                else:
                    logger.info("GPU Energy monitoring disabled in config.")
            else:
                logger.warning("NVML initialization failed. GPU monitoring disabled.")
        else:
            logger.info("Running on CPU, GPU monitoring disabled.")

        # Data Loading
        data_config = config.get("data", {})
        loader_config = config.get("data_loader", {})
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"),
            batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"),
            val_split=data_config.get("val_split", 0.1),
            seed=seed,
            num_workers=loader_config.get("num_workers", 4),
            pin_memory=(loader_config.get("pin_memory", True) and device.type == "cuda"),
            download=data_config.get("download", True),
        )
        logger.info("Dataloaders created.")

        # Model Instantiation
        model, input_adapter = get_model_and_adapter(config, device)
        model.to(device)
        logger.info(f"Model '{config.get('model', {}).get('name')}' on {device}.")
        try:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_total_params = sum(p.numel() for p in model.parameters())
            initial_metrics["model_parameters_trainable"] = num_params
            initial_metrics["model_parameters_total"] = num_total_params
            logger.info(f"Model trainable parameters: {num_params:,}")
            logger.info(f"Model total parameters: {num_total_params:,}")
        except Exception as e:
            logger.warning(f"Could not count model parameters: {e}")

        # FLOPs Profiling
        profiling_config = config.get("profiling", {})
        estimated_fwd_gflops = float('nan')
        estimated_bp_update_gflops = float('nan')
        if profiling_config.get("enabled", True):
            try:
                sample_input_img, _ = next(iter(train_loader))
                sample_input_device = sample_input_img.to(device)

                if input_adapter:
                    profile_input_constructor = lambda: input_adapter(sample_input_device[:1])
                    logger.debug("Using adapter and batch size 1 for FLOPs.")
                else:
                    profile_input_constructor = lambda: sample_input_device[:1]
                    logger.debug("No adapter, using batch size 1 for FLOPs.")

                logger.info("Profiling FLOPs...")
                # Call standard 'forward' for profiling
                fwd_gflops = profile_model_flops(
                    model,
                    profile_input_constructor,
                    device,
                    profiling_config.get("verbose", False)
                )

                if fwd_gflops is not None:
                    estimated_fwd_gflops = fwd_gflops
                    results["estimated_fwd_gflops"] = estimated_fwd_gflops
                    initial_metrics["estimated_fwd_gflops"] = estimated_fwd_gflops
                    logger.info(f"Estimated Forward Pass GFLOPs: {estimated_fwd_gflops:.4f} G")
                    if algorithm_name == "bp":
                        estimated_bp_update_gflops = estimated_fwd_gflops * 3.0 # Fwd + ~2x Fwd for Bwd
                        results["estimated_bp_update_gflops"] = estimated_bp_update_gflops
                        initial_metrics["estimated_bp_update_gflops"] = estimated_bp_update_gflops
                        logger.info(f"Estimated BP Update Cycle GFLOPs (Fwd+Bwd ~3x Fwd): {estimated_bp_update_gflops:.4f} G")
                    else:
                        results["estimated_bp_update_gflops"] = float('nan')
                        initial_metrics["estimated_bp_update_gflops"] = float('nan')
                        logger.info(f"Algorithm '{algorithm_name}' is BP-free, Est. BP Update GFLOPs is N/A.")
                else:
                    logger.warning("FLOPs profiling failed.")
                    results["estimated_fwd_gflops"] = float('nan')
                    initial_metrics["estimated_fwd_gflops"] = float('nan')
                    results["estimated_bp_update_gflops"] = float('nan')
                    initial_metrics["estimated_bp_update_gflops"] = float('nan')
            except StopIteration:
                logger.warning("Empty DataLoader, cannot perform FLOPs profiling.")
            except Exception as e:
                logger.error(f"FLOPs profiling failed: {e}", exc_info=True)

        # Log initial metrics
        if initial_metrics:
            step_ref[0] = 0 # Set step to 0 for initial logging
            metrics_to_log = {"global_step": step_ref[0], **initial_metrics}
            logger.debug(f"Logging initial metrics at global_step {step_ref[0]}: {initial_metrics.keys()}")
            log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        # --- Training ---
        logger.info("Starting training phase...")
        training_fn = get_training_function(algorithm_name)
        train_loop_start_time = time.time()
        total_energy_joules = None

        with monitor if monitor else contextlib.nullcontext() as active_monitor:
            training_args = {
                "model": model,
                "train_loader": train_loader,
                "config": config,
                "device": device,
                "wandb_run": wandb_run,
                "step_ref": step_ref,
                "gpu_handle": gpu_handle,
                "nvml_active": nvml_active
            }
            # Pass val_loader for algorithms that use it during training
            if algorithm_name in ["bp", "ff"]:
                training_args["val_loader"] = val_loader
            # Pass input_adapter for algorithms that expect it in the training function
            if algorithm_name in ["bp", "mf", "cafo"]:
                training_args["input_adapter"] = input_adapter

            # Call the appropriate training function
            train_output = training_fn(**training_args)

            # Store the returned peak memory if the function returns it
            if isinstance(train_output, (float, int)):
                peak_gpu_mem_during_training = train_output
                logger.info(f"Received Peak GPU Memory (sampled during training): {peak_gpu_mem_during_training:.2f} MiB")
            else:
                 logger.warning(f"Training function for '{algorithm_name}' returned type {type(train_output)}. Expected peak memory (float/int).")
                 peak_gpu_mem_during_training = float('nan') # Set to NaN if not returned

        # --- Post-Training Monitoring ---
        train_loop_duration = time.time() - train_loop_start_time
        results["training_duration_sec"] = train_loop_duration
        final_summary_step = step_ref[0] + 1 if step_ref[0] >= 0 else 0 # Handle case where step_ref might still be -1
        logger.debug(f"Training loop complete. Final global_step: {step_ref[0]}. Final summary step: {final_summary_step}")

        if monitor:
            total_energy_joules = monitor.get_total_energy_joules()
            if total_energy_joules is not None:
                results["total_gpu_energy_joules"] = total_energy_joules
                results["total_gpu_energy_wh"] = total_energy_joules / 3600.0
                logger.info(f"Total GPU Energy (Training): {total_energy_joules:.2f} J ({results['total_gpu_energy_wh']:.4f} Wh)")
            else:
                logger.warning("Energy monitoring failed to calculate total energy.")

        # Add the sampled peak memory to results
        results["peak_gpu_mem_used_mib"] = peak_gpu_mem_during_training

        # Log memory usage at the very end of the run
        if nvml_active and gpu_handle:
             mem_info_end = get_gpu_memory_usage(gpu_handle)
             if mem_info_end:
                 logger.info(f"GPU Mem (End): {mem_info_end[0]:.2f} MiB Used / {mem_info_end[1]:.2f} MiB Total")

        # --- Evaluation ---
        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get("evaluation", {})
        eval_criterion_name = eval_config.get("criterion", "CrossEntropyLoss")
        eval_criterion = nn.CrossEntropyLoss() if eval_criterion_name.lower() == "crossentropyloss" else None

        evaluation_fn = get_evaluation_function(algorithm_name)
        eval_results = {}
        test_loss_key = "test_loss"
        test_acc_key = "test_accuracy"

        eval_args = {
            "model": model,
            "data_loader": test_loader,
            "device": device
        }
        # Add arguments needed by specific evaluation functions
        if algorithm_name in ["bp", "mf", "cafo"]:
            eval_args["criterion"] = eval_criterion
        if algorithm_name in ["bp", "mf", "cafo"]:
            eval_args["input_adapter"] = input_adapter # FF eval uses internal logic
        if algorithm_name == "cafo":
            eval_args["predictors"] = getattr(model, "trained_predictors", None)
            eval_args["aggregation_method"] = config.get("algorithm_params", {}).get("aggregation_method", "sum")

        try:
            eval_output = evaluation_fn(**eval_args)
            if isinstance(eval_output, dict):
                eval_results[test_loss_key] = eval_output.get("eval_loss", float("nan"))
                eval_results[test_acc_key] = eval_output.get("eval_accuracy", float("nan"))
            elif isinstance(eval_output, tuple) and len(eval_output) == 2:
                eval_results[test_loss_key] = eval_output[0]
                eval_results[test_acc_key] = eval_output[1]
            else:
                logger.error(f"Unexpected eval return type: {type(eval_output)}")
                eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}
        except Exception as e:
            logger.error(f"Evaluation failed for {algorithm_name}: {e}", exc_info=True)
            eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}

        logger.info(f"Test Set Results: Acc: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}")
        results.update(eval_results)

    except Exception as e:
        logger.critical(f"\n--- Experiment Failed ---")
        logger.critical(f"Error during run: {e}", exc_info=True)
        results["error"] = str(e)
        if wandb_run and hasattr(wandb_run, 'finish'):
            try:
                wandb_run.finish(exit_code=1)
                logger.info("W&B run finished with error.")
            except Exception as e_wandb:
                logger.error(f"Error finishing W&B after exception: {e_wandb}")
        # Re-raise the exception for the main script to catch
        raise e
    finally:
        # --- Final Summary Logging (Single Call) ---
        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        final_summary_step = step_ref[0] + 1 if step_ref[0] >= -1 else 0 # Correct handling for step -1
        logger.debug(f"Final summary logging step: {final_summary_step}")

        final_summary_metrics = {
            "global_step": final_summary_step,
            "final/total_run_duration_sec": total_run_time,
            "final/peak_gpu_mem_used_mib": results.get("peak_gpu_mem_used_mib", float("nan")),
            "final/total_gpu_energy_joules": results.get("total_gpu_energy_joules", float("nan")),
            "final/total_gpu_energy_wh": results.get("total_gpu_energy_wh", float("nan")),
            "final/training_duration_sec": results.get("training_duration_sec", float("nan")),
            "final/Test_Accuracy": results.get("test_accuracy", float("nan")),
            "final/Test_Loss": results.get("test_loss", float("nan")),
            "final/estimated_fwd_gflops": results.get("estimated_fwd_gflops", float("nan")),
            "final/estimated_bp_update_gflops": results.get("estimated_bp_update_gflops", float('nan')),
        }

        log_metrics(final_summary_metrics, wandb_run=wandb_run, commit=True)

        logger.info(f"Total run duration: {format_time(total_run_time)}")

        # Clean up W&B run if it wasn't already finished due to error
        if wandb_run and hasattr(wandb_run, 'finish') and results.get("error") is None and os.environ.get("WANDB_MODE") != "offline":
            try:
                wandb_run.finish()
                logger.info("W&B run finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    return results