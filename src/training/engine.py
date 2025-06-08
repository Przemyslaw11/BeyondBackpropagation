import torch
import torch.nn as nn
import logging
import time
import os
import contextlib
import pynvml
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Callable, List
import pprint

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
from src.utils.codecarbon_utils import setup_codecarbon_tracker
from src.utils.profiling import profile_model_flops
from src.data_utils.datasets import get_dataloaders
from src.architectures import (
    FF_MLP, CaFo_CNN, MF_MLP
)
from src.algorithms import (
    get_training_function,
    get_evaluation_function,
)

def get_model_and_adapter(
    config: Dict[str, Any], device: torch.device
) -> Tuple[ nn.Module, Optional[Callable[[torch.Tensor], torch.Tensor]] ]:
    model_config = config.get("model", {})
    arch_name_raw = model_config.get("name", "")
    arch_name = arch_name_raw.lower()
    logger.debug(f"get_model_and_adapter: Raw arch name='{arch_name_raw}', Lowercase='{arch_name}'")

    arch_params = model_config.get("params", {})
    dataset_config = config.get("data", {})
    num_classes = dataset_config.get("num_classes", 10)
    input_channels = dataset_config.get("input_channels", 1)
    image_size = dataset_config.get("image_size", 28)

    algorithm_name = config.get("algorithm", {}).get("name", "").lower()
    is_bp_baseline = algorithm_name == "bp"

    logger.info(f"Getting architecture: {arch_name} (Algo: {algorithm_name.upper()})")

    if 'num_classes' not in arch_params: arch_params['num_classes'] = num_classes

    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    if arch_name in ['ff_mlp', 'mf_mlp']:
         if 'input_dim' not in arch_params:
             arch_params['input_dim'] = input_channels * image_size * image_size
             logger.debug(f"Calculated input_dim={arch_params['input_dim']} for {arch_name}")
         input_adapter = lambda x: x.view(x.shape[0], -1)
         logger.debug(f"Arch {arch_name} requires input flattening adapter.")
    elif arch_name == 'cafo_cnn':
         if 'input_channels' not in arch_params: arch_params['input_channels'] = input_channels
         if 'image_size' not in arch_params: arch_params['image_size'] = image_size
         input_adapter = None
         logger.debug(f"Arch {arch_name} does not require standard input adapter.")
    else:
        raise ValueError(f"Unknown model architecture category for name: {arch_name}. Expected 'ff_mlp', 'mf_mlp', or 'cafo_cnn'.")

    model: Optional[nn.Module] = None
    logger.debug(f"Checking architecture name '{arch_name}' against known types...")

    if arch_name in ['ff_mlp', 'mf_mlp']:
        logger.debug(f"'{arch_name}' matched MLP block.")
        if arch_name == 'ff_mlp':
            if is_bp_baseline:
                logger.info("Adapting modified FF_MLP structure for BP baseline -> Standard nn.Sequential MLP.")
                bp_input_dim = arch_params["input_dim"]
                hidden_dims = arch_params.get("hidden_dims", [])
                activation_name = arch_params.get("activation", "ReLU").lower()
                use_bias = arch_params.get("bias", True)
                if not hidden_dims: raise ValueError("BP baseline creation failed: hidden_dims missing for FF_MLP.")
                layers = []
                current_dim = bp_input_dim
                act_cls = nn.ReLU if activation_name == 'relu' else nn.Tanh
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
                    layers.append(act_cls())
                    current_dim = h_dim
                layers.append(nn.Linear(current_dim, num_classes, bias=use_bias))
                model = nn.Sequential(*layers)
                logger.debug("Created BP baseline Sequential model from modified FF_MLP spec.")
            else:
                model = FF_MLP(config=config, device=device, **arch_params)
                logger.debug("Using native modified FF_MLP structure.")
        elif arch_name == 'mf_mlp':
            model = MF_MLP(**arch_params)
            if is_bp_baseline: logger.info("Using standard forward pass of MF_MLP for BP baseline.")
            else: logger.debug("Using native MF_MLP structure.")

    elif arch_name == 'cafo_cnn':
        logger.debug(f"'{arch_name}' matched CNN block.")
        cnn_base = CaFo_CNN(**arch_params)

        if is_bp_baseline:
            logger.info(f"Creating BP baseline model from {arch_name.upper()} blocks + final Linear layer.")
            cnn_base.to(device)
            with torch.no_grad():
                 dummy_input_shape = (1, arch_params['input_channels'], arch_params['image_size'], arch_params['image_size'])
                 dummy_input = torch.randn(dummy_input_shape).to(device)
                 if hasattr(cnn_base, 'final_flat_dim'):
                     num_output_features = cnn_base.final_flat_dim
                     logger.debug(f"Got flattened output dimension from {arch_name.upper()} attribute: {num_output_features}")
                 else:
                     model_blocks = cnn_base.blocks
                     dummy_features = dummy_input
                     for block in model_blocks:
                          dummy_features = block(dummy_features)
                     num_output_features = dummy_features.view(dummy_features.size(0), -1).shape[1]
                     logger.debug(f"Dynamically computed flattened output dimension from {arch_name.upper()} blocks: {num_output_features}")
            cnn_base.cpu()

            model = nn.Sequential(
                *cnn_base.blocks,
                nn.Flatten(),
                nn.Linear(num_output_features, num_classes)
            )
            logger.debug(f"Created BP baseline Sequential model from {arch_name.upper()} spec.")
        else:
            model = cnn_base
            logger.debug(f"Using native {arch_name.upper()} structure.")
    else:
        logger.error(f"Architecture check failed inside factory. Lowercase name '{arch_name}' did not match ['ff_mlp', 'mf_mlp'] or ['cafo_cnn']. Check logic.")
        raise ValueError(f"Unknown model architecture: {arch_name}. Expected 'ff_mlp', 'mf_mlp', or 'cafo_cnn'.")

    if model is None: raise RuntimeError(f"Model instantiation failed for architecture: {arch_name}")
    logger.info(f"Model '{arch_name.upper()}' (Algo: {algorithm_name.upper()}) created.")
    return model, input_adapter

def run_training( config: Dict[str, Any], wandb_run: Optional[Any] = None ) -> Dict[str, Any]:
    results = {}
    nvml_active = False
    gpu_handle = None
    monitor = None
    tracker = None
    carbon_csv_path = None
    run_start_time = time.time()
    step_ref = [-1]
    peak_gpu_mem_during_training = float('nan')
    algorithm_name = config.get("algorithm", {}).get("name", "").lower()

    try:
        general_config = config.get("general", {})
        seed = general_config.get("seed", 42)
        env_seed_str = os.environ.get('EXPERIMENT_SEED')
        if env_seed_str is not None:
            try:
                env_seed = int(env_seed_str)
                logger.info(f"Found EXPERIMENT_SEED environment variable: {env_seed}. Overriding config seed ({seed}).")
                seed = env_seed
                general_config['seed'] = seed
            except ValueError:
                logger.warning(f"EXPERIMENT_SEED environment variable ('{env_seed_str}') is not a valid integer. Using config seed ({seed}).")

        set_seed(seed)
        logger.info(f"Using random seed: {seed}")

        device_preference = general_config.get("device", "auto").lower()
        if device_preference == "cuda" and torch.cuda.is_available(): device = torch.device("cuda")
        elif device_preference == "cpu": device = torch.device("cpu")
        else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} (Preference: '{device_preference}')")

        if wandb_run is None: wandb_run = setup_wandb(config, job_type="training")
        if wandb_run:
            try: wandb_run.define_metric("global_step", summary="max"); wandb_run.define_metric("*", step_metric="global_step")
            except Exception as e_define: logger.error(f"Failed to define W&B metric 'global_step': {e_define}")

        monitoring_config = config.get("monitoring", {})
        initial_metrics = {}
        if device.type == "cuda":
            if init_nvml():
                nvml_active = True; gpu_index = torch.cuda.current_device(); gpu_handle = get_gpu_handle(gpu_index)
                if gpu_handle:
                    logger.info(f"NVML active for GPU {gpu_index}.")
                    mem_info_start = get_gpu_memory_usage(gpu_handle)
                    if mem_info_start: initial_metrics["initial_gpu_mem_used_mib"] = mem_info_start[0]; logger.info(f"Initial GPU Mem: {mem_info_start[0]:.2f} MiB Used / {mem_info_start[1]:.2f} MiB Total")
                else: logger.warning(f"NVML active but failed get handle for GPU {gpu_index}."); nvml_active = False
                if monitoring_config.get("energy_enabled", True) and gpu_handle:
                    monitor = GPUEnergyMonitor(device_index=gpu_index, interval_sec=monitoring_config.get("energy_interval_sec", 0.2)); logger.info(f"GPU Energy monitor initialized (Interval: {monitor._interval_sec}s).")
                elif monitoring_config.get("energy_enabled", True): logger.warning("Energy monitoring enabled but failed to get GPU handle.")
                else: logger.info("GPU Energy monitoring disabled in config.")
            else: logger.warning("NVML initialization failed. GPU monitoring disabled.")
        else: logger.info("Running on CPU, GPU monitoring disabled.")

        tracker = setup_codecarbon_tracker(config, results)
        carbon_csv_path = results.get("codecarbon_csv_path")
        initial_metrics["codecarbon_enabled"] = results.get("codecarbon_enabled", False)
        initial_metrics["codecarbon_mode"] = results.get("codecarbon_mode", "N/A")
        initial_metrics["codecarbon_country_iso"] = results.get("codecarbon_country_iso", "N/A")

        data_config = config.get("data", {})
        loader_config = config.get("data_loader", {})
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_config.get("name", "FashionMNIST"), batch_size=loader_config.get("batch_size", 64),
            data_root=data_config.get("root", "./data"), val_split=data_config.get("val_split", 0.1),
            seed=seed,
            num_workers=loader_config.get("num_workers", 4),
            pin_memory=(loader_config.get("pin_memory", True) and device.type == "cuda"), download=data_config.get("download", True),
        )
        logger.info("Dataloaders created.")

        model, input_adapter = get_model_and_adapter(config, device); model.to(device)

        logger.info(f"Model '{config.get('model', {}).get('name')}' on {device}.")
        try:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); num_total_params = sum(p.numel() for p in model.parameters())
            initial_metrics["model_parameters_trainable"] = num_params; initial_metrics["model_parameters_total"] = num_total_params
            logger.info(f"Model trainable parameters: {num_params:,}"); logger.info(f"Model total parameters: {num_total_params:,}")
        except Exception as e: logger.warning(f"Could not count model parameters: {e}")

        profiling_config = config.get("profiling", {}); estimated_fwd_gflops = float('nan'); estimated_bp_update_gflops = float('nan')
        if profiling_config.get("enabled", True):
            try:
                sample_input_img, _ = next(iter(train_loader)); sample_input_device = sample_input_img.to(device)
                profile_input_constructor = (lambda: input_adapter(sample_input_device[:1])) if input_adapter else (lambda: sample_input_device[:1])
                logger.info("Profiling FLOPs...")
                fwd_gflops = profile_model_flops(model, profile_input_constructor, device, profiling_config.get("verbose", False))
                if fwd_gflops is not None:
                    estimated_fwd_gflops = fwd_gflops; results["estimated_fwd_gflops"] = estimated_fwd_gflops; initial_metrics["estimated_fwd_gflops"] = estimated_fwd_gflops
                    logger.info(f"Estimated Forward Pass GFLOPs: {estimated_fwd_gflops:.4f} G")
                    if algorithm_name == "bp":
                        estimated_bp_update_gflops = estimated_fwd_gflops * 3.0; results["estimated_bp_update_gflops"] = estimated_bp_update_gflops; initial_metrics["estimated_bp_update_gflops"] = estimated_bp_update_gflops
                        logger.info(f"Estimated BP Update Cycle GFLOPs (Fwd+Bwd ~3x Fwd): {estimated_bp_update_gflops:.4f} G")
                    else:
                        results["estimated_bp_update_gflops"] = float('nan'); initial_metrics["estimated_bp_update_gflops"] = float('nan'); logger.info(f"Algorithm '{algorithm_name}' is BP-free, Est. BP Update GFLOPs is N/A.")
                else:
                    logger.warning("FLOPs profiling failed."); results["estimated_fwd_gflops"] = float('nan'); initial_metrics["estimated_fwd_gflops"] = float('nan'); results["estimated_bp_update_gflops"] = float('nan'); initial_metrics["estimated_bp_update_gflops"] = float('nan')
            except StopIteration: logger.warning("Empty DataLoader, cannot perform FLOPs profiling.")
            except Exception as e: logger.error(f"FLOPs profiling failed: {e}", exc_info=True)

        if initial_metrics:
            step_ref[0] = 0; metrics_to_log = {"global_step": step_ref[0], **initial_metrics}
            logger.debug(f"Logging initial metrics at global_step {step_ref[0]}: {initial_metrics.keys()}"); log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        logger.info("Starting training phase...")
        training_fn = get_training_function(algorithm_name)
        train_loop_start_time = time.time()

        with monitor if monitor else contextlib.nullcontext() as active_monitor:
            training_args = {
                "model": model, "train_loader": train_loader, "config": config, "device": device,
                "wandb_run": wandb_run, "step_ref": step_ref, "gpu_handle": gpu_handle, "nvml_active": nvml_active
            }
            if algorithm_name in ["bp", "ff", "cafo", "mf"]:
                training_args["val_loader"] = val_loader

            if (algorithm_name == "bp" and not isinstance(model, CaFo_CNN)) or \
               (algorithm_name == "mf"):
                 training_args["input_adapter"] = input_adapter

            train_output = training_fn(**training_args)

            if isinstance(train_output, (float, int)): peak_gpu_mem_during_training = train_output; logger.info(f"Received Peak GPU Memory (sampled during training): {peak_gpu_mem_during_training:.2f} MiB")
            else: logger.warning(f"Training function for '{algorithm_name}' returned type {type(train_output)}. Expected peak memory (float/int). Peak memory set to NaN."); peak_gpu_mem_during_training = float('nan')

        train_loop_duration = time.time() - train_loop_start_time
        results["training_duration_sec"] = train_loop_duration
        logger.debug(f"Training loop complete. Final global_step: {step_ref[0]}.")
        results["peak_gpu_mem_used_mib"] = peak_gpu_mem_during_training

        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get("evaluation", {}); eval_criterion_name = eval_config.get("criterion", "CrossEntropyLoss")
        eval_criterion = nn.CrossEntropyLoss() if eval_criterion_name.lower() == "crossentropyloss" else None
        evaluation_fn = get_evaluation_function(algorithm_name)
        eval_results = {}; test_loss_key = "test_loss"; test_acc_key = "test_accuracy"
        eval_args = { "model": model, "data_loader": test_loader, "device": device }
        if algorithm_name in ["bp", "cafo", "mf"]: eval_args["criterion"] = eval_criterion

        if (algorithm_name == "bp" and not isinstance(model, CaFo_CNN)) or \
           (algorithm_name == "mf"):
             eval_args["input_adapter"] = input_adapter

        if algorithm_name == "cafo": eval_args["predictors"] = getattr(model, "trained_predictors", None); eval_args["aggregation_method"] = config.get("algorithm_params", {}).get("aggregation_method", "sum")
        try:
            eval_output = evaluation_fn(**eval_args)
            if isinstance(eval_output, dict): eval_results[test_loss_key] = eval_output.get("eval_loss", float("nan")); eval_results[test_acc_key] = eval_output.get("eval_accuracy", float("nan"))
            elif isinstance(eval_output, tuple) and len(eval_output) == 2: eval_results[test_loss_key] = eval_output[0]; eval_results[test_acc_key] = eval_output[1]
            else: logger.error(f"Unexpected eval return type: {type(eval_output)}. Setting results to NaN."); eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}
        except Exception as e: logger.error(f"Evaluation failed for {algorithm_name}: {e}", exc_info=True); eval_results = {test_loss_key: float("nan"), test_acc_key: float("nan")}
        logger.info(f"Test Set Results: Acc: {eval_results.get(test_acc_key, 'N/A'):.2f}%, Loss: {eval_results.get(test_loss_key, 'N/A'):.4f}"); results.update(eval_results)

    except Exception as e:
        logger.critical(f"\n--- Experiment Failed ---"); logger.critical(f"Error during run: {e}", exc_info=True)
        results["error"] = str(e)
        if wandb_run and hasattr(wandb_run, 'finish'):
            try: wandb_run.finish(exit_code=1); logger.info("W&B run finished with error.")
            except Exception as e_wandb: logger.error(f"Error finishing W&B after exception: {e_wandb}")
    finally:
        total_run_time = time.time() - run_start_time
        results["total_run_duration_sec"] = total_run_time
        final_summary_step = step_ref[0] + 1 if step_ref[0] >= -1 else 0
        logger.debug(f"Final summary logging step: {final_summary_step}")
        codecarbon_emissions_kg = float('nan'); codecarbon_emissions_g = float('nan')
        total_energy_joules = results.get("total_gpu_energy_joules")
        if tracker:
            logger.info("Stopping CodeCarbon tracker...")
            try:
                emissions_data = tracker.stop()
                if emissions_data is not None:
                     logger.info(f"CodeCarbon tracker stopped. Attempting to read from: {carbon_csv_path}")
                else:
                     logger.warning("CodeCarbon tracker.stop() returned None. Attempting CSV read.")

                if carbon_csv_path and os.path.exists(carbon_csv_path):
                    max_retries = 5; retry_delay = 0.5
                    for attempt in range(max_retries):
                        try:
                            time.sleep(retry_delay)
                            df = pd.read_csv(carbon_csv_path)
                            if not df.empty and 'emissions' in df.columns:
                                emissions_value_kg = df['emissions'].iloc[-1]
                                if pd.notna(emissions_value_kg):
                                    codecarbon_emissions_kg = float(emissions_value_kg); codecarbon_emissions_g = codecarbon_emissions_kg * 1000.0
                                    logger.info(f"Read emissions: {codecarbon_emissions_kg:.6f} kgCO2e ({codecarbon_emissions_g:.3f} gCO2e) (attempt {attempt+1})"); break
                                else: logger.warning(f"Read CSV attempt {attempt+1}, 'emissions' value NaN.")
                            elif not df.empty: logger.warning(f"Read CSV attempt {attempt+1}, 'emissions' column missing.")
                            else: logger.warning(f"Read CSV attempt {attempt+1}, file empty.")
                        except pd.errors.EmptyDataError: logger.warning(f"Read CSV attempt {attempt+1} failed: File empty.")
                        except FileNotFoundError: logger.error(f"Read CSV attempt {attempt+1} failed: File disappeared {carbon_csv_path}."); break
                        except Exception as e_read: logger.error(f"Read CSV attempt {attempt+1} failed: {e_read}", exc_info=True)
                        if attempt < max_retries - 1: logger.info(f"Retrying CSV read in {retry_delay}s...")
                        else: logger.error(f"Failed to read emissions from {carbon_csv_path} after {max_retries} attempts.")
                elif carbon_csv_path: logger.warning(f"CodeCarbon CSV path '{carbon_csv_path}' does not exist.")
                else: logger.warning("CodeCarbon CSV path not set.")
            except Exception as e_tracker: logger.error(f"Error stopping/reading CodeCarbon tracker: {e_tracker}", exc_info=True)
        results["codecarbon_emissions_kgCO2e"] = codecarbon_emissions_kg; results["codecarbon_emissions_gCO2e"] = codecarbon_emissions_g
        if monitor and total_energy_joules is None :
            total_energy_joules = monitor.stop()
        if total_energy_joules is not None:
            results["total_gpu_energy_joules"] = total_energy_joules
            results["total_gpu_energy_wh"] = total_energy_joules / 3600.0
            logger.info(f"Total GPU Energy (NVML Monitor): {total_energy_joules:.2f} J ({results['total_gpu_energy_wh']:.4f} Wh)")
        elif monitor: logger.warning("Energy monitoring failed to calculate total energy.")

        if nvml_active and gpu_handle:
             mem_info_end = get_gpu_memory_usage(gpu_handle)
             if mem_info_end: logger.info(f"GPU Mem (End): {mem_info_end[0]:.2f} MiB Used / {mem_info_end[1]:.2f} MiB Total")
             else: logger.warning("Failed to get final GPU memory usage.")
        final_summary_metrics = {
            "global_step": final_summary_step, "final/total_run_duration_sec": total_run_time, "final/peak_gpu_mem_used_mib": results.get("peak_gpu_mem_used_mib", float("nan")),
            "final/total_gpu_energy_joules": results.get("total_gpu_energy_joules", float("nan")), "final/total_gpu_energy_wh": results.get("total_gpu_energy_wh", float("nan")),
            "final/training_duration_sec": results.get("training_duration_sec", float("nan")), "final/Test_Accuracy": results.get("test_accuracy", float("nan")),
            "final/Test_Loss": results.get("test_loss", float("nan")), "final/estimated_fwd_gflops": results.get("estimated_fwd_gflops", float("nan")),
            "final/estimated_bp_update_gflops": results.get("estimated_bp_update_gflops", float('nan')), "final/codecarbon_emissions_gCO2e": codecarbon_emissions_g,
        }
        log_metrics(final_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.info(f"Total run duration: {format_time(total_run_time)}")
        if pd.notna(results.get("codecarbon_emissions_gCO2e")): logger.info(f"--> Final Calculated Emissions: {results['codecarbon_emissions_gCO2e']:.3f} gCO2e")
        else: logger.info("--> Final Calculated Emissions: Not Available (NaN)")
        if wandb_run and hasattr(wandb_run, 'finish') and results.get("error") is None:
            if os.environ.get("WANDB_MODE", "").lower() == "offline": logger.info("W&B running in offline mode. Run 'wandb sync' later.")
            else:
                try: wandb_run.finish(); logger.info("W&B run finished.")
                except Exception as e: logger.error(f"Error finishing W&B run: {e}")
        if nvml_active: shutdown_nvml()

    results.pop('codecarbon_emissions_kgCO2e', None)
    return results