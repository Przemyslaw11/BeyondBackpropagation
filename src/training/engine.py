# File: src/training/engine.py
import torch
import torch.nn as nn
import logging
import time
import os
import contextlib
import pynvml
from typing import Dict, Any, Optional, Tuple, Callable, List # Added List

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
)

# --- get_model_and_adapter function (no changes needed here) ---
def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device
) -> Tuple[
    nn.Module, Optional[Callable[[torch.Tensor], torch.Tensor]]
]:
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

    model: Optional[nn.Module] = None
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    logger.info(
        f"Creating model architecture: {arch_name} (BP Baseline: {is_bp_baseline})"
    )

    # --- Determine necessary parameters and adapter based on architecture ---
    if arch_name in ["ff_mlp", "mf_mlp"]:
        if "input_dim" not in arch_params:
            arch_params["input_dim"] = input_channels * image_size * image_size
            logger.debug(f"Calculated input_dim for MLP: {arch_params['input_dim']}")
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        input_adapter = lambda x: x.view(x.shape[0], -1) # Flatten input
        logger.debug(f"Architecture {arch_name} requires input flattening adapter.")
    elif arch_name == "cafo_cnn":
        if "input_channels" not in arch_params:
            arch_params["input_channels"] = input_channels
        if "image_size" not in arch_params:
            arch_params["image_size"] = image_size
        if "num_classes" not in arch_params:
            arch_params["num_classes"] = num_classes
        input_adapter = None # CNNs typically don't need flattening adapter here
        logger.debug(f"Architecture {arch_name} does not require standard input adapter.")
    else:
        raise ValueError(f"Unknown model architecture name in config: {arch_name}")

    # --- Instantiate based on architecture name ---
    if arch_name == "ff_mlp":
        ff_model_instance = FF_MLP(**arch_params)
        if is_bp_baseline:
            logger.info("Adapting FF_MLP structure for BP baseline -> Standard nn.Sequential MLP.")
            layers = []
            bp_input_dim = arch_params["input_dim"]
            layers.append(nn.Linear(bp_input_dim, ff_model_instance.input_adapter_layer.out_features,
                                    bias=ff_model_instance.input_adapter_layer.bias is not None))
            layers.append(type(ff_model_instance.first_layer_activation)())
            current_dim = ff_model_instance.input_adapter_layer.out_features
            for ff_layer in ff_model_instance.layers:
                layers.append(nn.Linear(current_dim, ff_layer.linear.out_features,
                                        bias=ff_layer.linear.bias is not None))
                layers.append(type(ff_layer.activation)())
                current_dim = ff_layer.linear.out_features
            layers.append(nn.Linear(current_dim, num_classes))
            model = nn.Sequential(*layers)
            logger.debug("Created BP baseline Sequential model from FF_MLP spec.")
        else:
            model = ff_model_instance
            logger.debug("Using native FF_MLP structure.")
    elif arch_name == "cafo_cnn":
        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
            logger.info("Creating BP baseline model from CaFo_CNN blocks + final Linear layer.")
            cafo_base.to(device) # Move temporarily for shape calc
            with torch.no_grad():
                # Create dummy input directly on device
                dummy_input = torch.randn(1, input_channels, image_size, image_size).to(device)
                last_block_output = cafo_base.forward_blocks_only(dummy_input)
                num_output_features = last_block_output.numel() # Flattened size
            cafo_base.cpu() # Move back if needed elsewhere
            logger.debug(f"Flattened output dimension from CaFo blocks: {num_output_features}")
            model = nn.Sequential(cafo_base.blocks, nn.Flatten(), nn.Linear(num_output_features, num_classes))
            logger.debug("Created BP baseline Sequential model from CaFo_CNN spec.")
        else:
            model = cafo_base
            logger.debug("Using native CaFo_CNN structure.")
    elif arch_name == "mf_mlp":
        model = MF_MLP(**arch_params) # MF_MLP handles BP mode via standard forward
        if is_bp_baseline:
            logger.info("Using standard forward pass of MF_MLP for BP baseline.")
        else:
            logger.debug("Using native MF_MLP structure (training uses specific forward methods).")

    if model is None:
        raise RuntimeError(f"Failed to instantiate model for architecture: {arch_name}")

    logger.info(f"Model '{arch_name}' (BP: {is_bp_baseline}) and input adapter (type: {type(input_adapter)}) created.")
    return model, input_adapter


# --- run_training (MODIFIED) ---
def run_training(
    config: Dict[str, Any], wandb_run: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Runs the full training and evaluation pipeline for one experiment configuration.
    MODIFIED: Uses define_metric and passes step_ref.
    """
    results = {}
    nvml_active = False
    gpu_handle = None
    monitor = None
    run_start_time = time.time()
    # MODIFIED: Use a list to pass step by reference
    step_ref = [-1] # Start at -1 so first increment makes it 0

    try:
        # --- Setup ---
        general_config = config.get("general", {})
        seed = general_config.get("seed", 42)
        set_seed(seed)
        logger.info(f"Using random seed: {seed}")

        # --- Device Setup ---
        device_preference = general_config.get("device", "auto").lower()
        if device_preference == "cuda" and torch.cuda.is_available(): device = torch.device("cuda")
        elif device_preference == "cpu": device = torch.device("cpu")
        else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} (Preference: '{device_preference}')")

        # --- W&B Initialization ---
        if wandb_run is None:
            wandb_run = setup_wandb(config, job_type="training")

        # --- Define W&B Metric AFTER init --- *** MODIFIED ***
        if wandb_run:
            try:
                # Define "global_step" as the primary x-axis.
                wandb_run.define_metric("global_step", summary="max") # Track the max step reached
                # Define that all other metrics use "global_step" as their x-axis.
                wandb_run.define_metric("*", step_metric="global_step")
                logger.info("Defined 'global_step' as the default x-axis metric for W&B.")
            except Exception as e_define:
                logger.error(f"Failed to define W&B metric 'global_step': {e_define}")
        # --- End W&B Metric Definition ---

        # --- Monitoring Initialization ---
        monitoring_config = config.get("monitoring", {})
        initial_metrics = {} # Accumulate initial metrics
        if device.type == "cuda":
            if init_nvml():
                nvml_active = True
                gpu_index = torch.cuda.current_device()
                gpu_handle = get_gpu_handle(gpu_index)
                if gpu_handle:
                    logger.info(f"NVML active for GPU {gpu_index}.")
                    mem_info_start = get_gpu_memory_usage(gpu_handle)
                    if mem_info_start: initial_metrics["initial_gpu_mem_used_mib"] = mem_info_start[0]
                    if monitoring_config.get("energy_enabled", True):
                        monitor = GPUEnergyMonitor(
                            device_index=gpu_index,
                            interval_sec=monitoring_config.get("energy_interval_sec", 0.2),
                        ); logger.info(f"GPU Energy monitor initialized (Interval: {monitor._interval_sec}s).")
                    else: logger.info("GPU Energy monitoring disabled.")
                else: logger.warning(f"NVML active but failed get handle for GPU {gpu_index}."); nvml_active = False
            else: logger.warning("NVML initialization failed. GPU monitoring disabled.")
        else: logger.info("Running on CPU, GPU monitoring disabled.")

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
            pin_memory=(loader_config.get("pin_memory", True) if device.type == "cuda" else False),
            download=data_config.get("download", True),
        )
        logger.info("Dataloaders created.")

        # --- Model Instantiation ---
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
        except Exception as e: logger.warning(f"Could not count model parameters: {e}")

        # --- FLOPs Profiling ---
        profiling_config = config.get("profiling", {})
        if profiling_config.get("enabled", True):
            try:
                sample_input_img, _ = next(iter(train_loader))
                sample_input_device = sample_input_img.to(device)
                # --- MODIFIED: Define input constructor to use batch size 1 ---
                if input_adapter:
                    # Apply adapter first, then slice
                    profile_input_constructor = lambda: input_adapter(sample_input_device[:1]) # Slice after adapter
                    logger.debug("Using adapter and batch size 1 for FLOPs.")
                else:
                    profile_input_constructor = lambda: sample_input_device[:1] # Slice directly
                    logger.debug("No adapter, using batch size 1 for FLOPs.")
                # --- END MODIFICATION ---

                logger.info("Profiling FLOPs...")
                gmacs = profile_model_flops(model, profile_input_constructor, device, profiling_config.get("verbose", False))
                if gmacs is not None:
                    results["estimated_gmacs"] = gmacs; results["estimated_gflops"] = gmacs * 2.0
                    initial_metrics["estimated_gmacs"] = gmacs
                    initial_metrics["estimated_gflops"] = results["estimated_gflops"]
                    logger.info(f"Estimated GFLOPs (2*GMACs): {results['estimated_gflops']:.4f} G")
                else: logger.warning("FLOPs profiling failed.")
            except StopIteration: logger.warning("Empty DataLoader for FLOPs profiling.")
            except Exception as e: logger.error(f"FLOPs profiling failed: {e}", exc_info=True)

        # --- Log all initial metrics at step 0 --- *** MODIFIED ***
        if initial_metrics:
             step_ref[0] = 0 # Explicitly set step 0
             metrics_to_log = {"global_step": step_ref[0], **initial_metrics}
             logger.debug(f"Logging initial metrics at global_step {step_ref[0]}: {initial_metrics.keys()}")
             log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        # --- Training ---
        logger.info("Starting training phase...")
        algorithm_name = config.get("algorithm", {}).get("name", "").lower()
        training_fn = get_training_function(algorithm_name)

        peak_gpu_mem = float("nan")
        train_loop_start_time = time.time()
        total_energy_joules = None

        # *** MODIFIED: Pass step_ref list ***
        with monitor if monitor else contextlib.nullcontext() as active_monitor:
            training_args = {
                "model": model, "train_loader": train_loader, "config": config,
                "device": device, "wandb_run": wandb_run,
                "step_ref": step_ref # Pass the list reference
            }
            # Add algorithm specific args
            if algorithm_name == "bp":
                training_args["val_loader"] = val_loader
                training_args["input_adapter"] = input_adapter
            elif algorithm_name in ["mf", "ff", "cafo"]:
                # Add input adapter if needed by these algos
                training_args["input_adapter"] = input_adapter

            # Call the training function
            training_fn(**training_args)

            # Retrieve the final step count (already updated in step_ref)

        # --- Post-Training Monitoring ---
        train_loop_duration = time.time() - train_loop_start_time
        results["training_duration_sec"] = train_loop_duration
        final_summary_step = step_ref[0] + 1 # Step for final summary *** MODIFIED ***
        logger.debug(f"Training loop complete. Final global_step: {step_ref[0]}. Final summary step: {final_summary_step}")

        if monitor:
            total_energy_joules = monitor.get_total_energy_joules()
            if total_energy_joules is not None:
                results["total_gpu_energy_joules"] = total_energy_joules
                results["total_gpu_energy_wh"] = total_energy_joules / 3600.0
                logger.info(f"Total GPU Energy (Training): {total_energy_joules:.2f} J ({results['total_gpu_energy_wh']:.4f} Wh)")
            else: logger.warning("Energy monitoring failed to calculate total energy.")

        if nvml_active and gpu_handle:
            time.sleep(0.1) # Short delay
            try:
                peak_gpu_mem = float('nan')
                try:
                    # Try nvmlDeviceGetMaxMemoryUsage first
                    peak_mem_info = pynvml.nvmlDeviceGetMaxMemoryUsage(gpu_handle)
                    peak_gpu_mem_nvml = peak_mem_info / (1024**2)
                    logger.info(f"Peak GPU Memory Usage (NVML Max): {peak_gpu_mem_nvml:.2f} MiB")
                    peak_gpu_mem = peak_gpu_mem_nvml
                except (pynvml.NVMLError_NotSupported, AttributeError):
                    # Fallback: Use current memory usage at the end as proxy
                    logger.warning("Direct peak memory query not supported.")
                    mem_info_end = get_gpu_memory_usage(gpu_handle)
                    if mem_info_end:
                        peak_gpu_mem = mem_info_end[0]
                        logger.warning(f"Using mem at end as proxy for peak: {peak_gpu_mem:.2f} MiB")
                    else:
                        logger.error("Failed to get end-of-run memory usage as fallback.")
                        peak_gpu_mem = float('nan')
                except pynvml.NVMLError as e_nvml:
                    logger.error(f"NVML Error getting peak memory: {e_nvml}")
                    peak_gpu_mem = float('nan')
                except Exception as e_mem_peak:
                    logger.error(f"Unexpected error getting peak memory: {e_mem_peak}")
                    peak_gpu_mem = float('nan')

                # Use torch.tensor for robust isnan check
                if not torch.isnan(torch.tensor(peak_gpu_mem)):
                    results["peak_gpu_mem_used_mib"] = peak_gpu_mem
                else:
                    results["peak_gpu_mem_used_mib"] = float("nan")
                    logger.warning("Peak GPU memory value is NaN, recording NaN.")

                mem_info_end = get_gpu_memory_usage(gpu_handle)
                if mem_info_end: logger.info(f"GPU Mem (End): {mem_info_end[0]:.2f} MiB Used / {mem_info_end[1]:.2f} MiB Total")

            except Exception as e_mem: logger.error(f"Failed to get memory usage: {e_mem}", exc_info=True)


        # --- Evaluation ---
        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get("evaluation", {})
        eval_criterion_name = eval_config.get("criterion", "CrossEntropyLoss")
        eval_criterion = nn.CrossEntropyLoss() if eval_criterion_name.lower() == "crossentropyloss" else None

        evaluation_fn = get_evaluation_function(algorithm_name)
        eval_results = {}
        test_loss_key = "test_loss"
        test_acc_key = "test_accuracy"

        eval_args = {"model": model, "data_loader": test_loader, "device": device}
        if algorithm_name in ["bp", "mf", "cafo"]: eval_args["criterion"] = eval_criterion
        if algorithm_name in ["bp", "mf", "cafo"]: eval_args["input_adapter"] = input_adapter
        if algorithm_name == "cafo":
            eval_args["predictors"] = getattr(model, "trained_predictors", None)
            eval_args["aggregation_method"] = config.get("algorithm_params", {}).get("aggregation_method", "sum")

        try:
            eval_output = evaluation_fn(**eval_args)
            if isinstance(eval_output, dict):
                eval_results[test_loss_key] = eval_output.get("eval_loss", float("nan"))
                eval_results[test_acc_key] = eval_output.get("eval_accuracy", float("nan"))
            elif isinstance(eval_output, tuple) and len(eval_output) == 2:
                eval_results[test_loss_key] = eval_output[0]; eval_results[test_acc_key] = eval_output[1]
            else:
                logger.error(f"Unexpected eval return type: {type(eval_output)}")
                eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}
        except Exception as e:
            logger.error(f"Evaluation failed for {algorithm_name}: {e}", exc_info=True)
            eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}

        logger.info(f"Test Set Results: Acc: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}")
        results.update(eval_results) # Add test results to main results dict

    except Exception as e:
        logger.critical(f"\n--- Experiment Failed ---")
        logger.critical(f"Error during run: {e}", exc_info=True)
        results["error"] = str(e)
        if wandb_run and hasattr(wandb_run, 'finish'): # Check if finish method exists
            try: wandb_run.finish(exit_code=1); logger.info("W&B run finished with error.")
            except Exception as e_wandb: logger.error(f"Error finishing W&B after exception: {e_wandb}")
        # Re-raise the exception after attempting to finish W&B run
        raise e
    finally:
        # --- Final Summary Logging (Single Call) --- *** MODIFIED ***
        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        # Use the final_summary_step calculated after training loop
        logger.debug(f"Final summary logging step: {final_summary_step}")

        final_summary_metrics = {
            "global_step": final_summary_step, # Add step here
            "final/total_run_duration_sec": total_run_time,
            "final/peak_gpu_mem_used_mib": results.get("peak_gpu_mem_used_mib", float("nan")),
            "final/total_gpu_energy_joules": results.get("total_gpu_energy_joules", float("nan")),
            "final/total_gpu_energy_wh": results.get("total_gpu_energy_wh", float("nan")),
            "final/training_duration_sec": results.get("training_duration_sec", float("nan")),
            "final/Test_Accuracy": results.get("test_accuracy", float("nan")),
            "final/Test_Loss": results.get("test_loss", float("nan")),
            "final/estimated_gflops": results.get("estimated_gflops", float("nan")),
            "final/estimated_gmacs": results.get("estimated_gmacs", float("nan")),
        }
        # Filter out potential NaN values before logging to W&B if needed
        # final_summary_metrics_clean = {k: v for k, v in final_summary_metrics.items() if not (isinstance(v, float) and torch.isnan(torch.tensor(v)))}

        # Log all final summary metrics in a single call
        log_metrics(final_summary_metrics, wandb_run=wandb_run, commit=True)

        logger.info(f"Total run duration: {format_time(total_run_time)}")
        # NVML shutdown handled by atexit

        if wandb_run and hasattr(wandb_run, 'finish') and results.get("error") is None and os.environ.get("WANDB_MODE") != "offline":
            try:
                wandb_run.finish() # Finish should commit automatically
                logger.info("W&B run finished.")
            except Exception as e: logger.error(f"Error finishing W&B run: {e}")

    return results