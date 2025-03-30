import torch
import torch.nn as nn
import logging
import time
import os
from typing import Dict, Any, Optional, Tuple, Callable

# Import necessary components from our project structure
from src.utils.config_parser import load_config # May not be needed here if config is passed in
from src.utils.helpers import set_seed, create_directory_if_not_exists, format_time
from src.utils.logging_utils import setup_wandb, log_metrics, logger # Use the configured logger
from src.utils.monitoring import init_nvml, shutdown_nvml, get_gpu_handle, get_gpu_memory_usage, get_gpu_power_usage # Add power later for energy
from src.utils.profiling import profile_model_flops
from src.data_utils.datasets import get_dataloaders
from src.architectures import FF_MLP, CaFo_CNN, MF_MLP # Import base architectures
# Import training/evaluation functions for algorithms and baselines
from src.algorithms import train_ff_model, evaluate_ff_model # Assuming these will be implemented fully
from src.algorithms import train_cafo_model, evaluate_cafo_model
from src.algorithms import train_mf_model, evaluate_mf_model
from src.baselines import train_bp_model, evaluate_bp_model

# Placeholder for FF training/eval functions if not fully implemented yet
if 'train_ff_model' not in globals():
    def train_ff_model(*args, **kwargs):
        logger.error("train_ff_model is not fully implemented in src/algorithms/ff.py")
        raise NotImplementedError("train_ff_model not implemented")
if 'evaluate_ff_model' not in globals():
     def evaluate_ff_model(*args, **kwargs):
        logger.error("evaluate_ff_model is not fully implemented in src/algorithms/ff.py")
        raise NotImplementedError("evaluate_ff_model not implemented")


def get_model_and_adapter(config: Dict[str, Any]) -> Tuple[nn.Module, Optional[Callable]]:
    """
    Instantiates the model based on the configuration and returns an optional input adapter.

    Handles specific adaptations needed for BP baselines (e.g., adding a classifier).
    """
    model_config = config.get('model', {})
    arch_name = model_config.get('name', '').lower()
    arch_params = model_config.get('params', {})
    dataset_config = config.get('dataset', {})
    num_classes = dataset_config.get('num_classes', 10) # Default, should be in config
    input_channels = dataset_config.get('input_channels', 1) # Default, should be in config
    image_size = dataset_config.get('image_size', 28) # Default, should be in config

    algorithm_name = config.get('algorithm', {}).get('name', '').lower()
    is_bp_baseline = algorithm_name == 'bp'

    model = None
    input_adapter = None # Function to adapt input, e.g., flatten

    logger.info(f"Creating model architecture: {arch_name} with params: {arch_params}")
    logger.info(f"Input: {input_channels} channels, {image_size}x{image_size}, Num classes: {num_classes}")

    if arch_name == 'ff_mlp':
        # FF_MLP expects flattened input dimension in params
        if 'input_dim' not in arch_params:
             arch_params['input_dim'] = input_channels * image_size * image_size
        if 'num_classes' not in arch_params:
             arch_params['num_classes'] = num_classes
        model = FF_MLP(**arch_params)
        # FF training handles input prep internally, BP baseline needs adapter
        if is_bp_baseline:
             # BP baseline for FF: Use the layers but train end-to-end.
             # Need a standard MLP structure. FF_MLP isn't directly suitable for BP.
             # We should define a separate SimpleMLP or adapt FF_MLP structure.
             # For now, let's raise an error indicating a dedicated BP MLP is needed.
             # raise NotImplementedError("BP Baseline for FF_MLP requires a standard MLP architecture definition (e.g., SimpleMLP).")
             # --- OR --- Adapt FF_MLP for BP (less ideal, removes FF specifics):
             logger.warning("Adapting FF_MLP for BP baseline. This might not be the intended fair comparison architecture.")
             # Create a standard sequential model using FF_Layer linear parts + final layer
             layers = []
             current_dim = arch_params['input_dim'] # BP uses raw input dim
             for ff_layer in model.layers: # model is the FF_MLP instance
                 layers.append(ff_layer.linear) # Use only the linear part
                 layers.append(ff_layer.activation) # Use the activation
                 current_dim = ff_layer.linear.out_features
             layers.append(nn.Linear(current_dim, num_classes)) # Add final BP classifier
             model = nn.Sequential(*layers)
             input_adapter = lambda x: x.view(x.shape[0], -1) # Flatten input for MLP

    elif arch_name == 'cafo_cnn':
        if 'input_channels' not in arch_params:
             arch_params['input_channels'] = input_channels
        if 'image_size' not in arch_params:
             arch_params['image_size'] = image_size
        if 'num_classes' not in arch_params:
             arch_params['num_classes'] = num_classes

        cafo_base = CaFo_CNN(**arch_params)
        if is_bp_baseline:
             # BP baseline: Use CaFo blocks + Flatten + new Linear classifier
             logger.info("Creating BP baseline model from CaFo_CNN blocks.")
             with torch.no_grad():
                  dummy_input = torch.randn(1, input_channels, image_size, image_size)
                  last_block_output = cafo_base.forward_blocks_only(dummy_input)
                  num_output_features = last_block_output.numel()
             model = nn.Sequential(
                  cafo_base.blocks, # Use the ModuleList of blocks
                  nn.Flatten(),
                  nn.Linear(num_output_features, num_classes)
             )
             # No input adapter needed, CNN handles spatial input
        else:
             # Use the CaFo model directly for CaFo training
             model = cafo_base

    elif arch_name == 'mf_mlp':
        if 'input_dim' not in arch_params:
             arch_params['input_dim'] = input_channels * image_size * image_size
        if 'num_classes' not in arch_params:
             arch_params['num_classes'] = num_classes
        model = MF_MLP(**arch_params)
        # MF_MLP standard forward expects flattened input
        input_adapter = lambda x: x.view(x.shape[0], -1)

    else:
        raise ValueError(f"Unknown model architecture name: {arch_name}")

    if model is None:
         raise RuntimeError(f"Failed to instantiate model for architecture: {arch_name}")

    return model, input_adapter


def run_training(config: Dict[str, Any], wandb_run: Optional[Any] = None) -> Dict[str, Any]:
    """
    Runs the full training and evaluation pipeline for one experiment configuration.

    Args:
        config: Dictionary containing the experiment configuration.
        wandb_run: Initialized Weights & Biases run object (optional).

    Returns:
        Dictionary containing key results (e.g., test accuracy, time, energy).
    """
    results = {}
    nvml_active = False
    gpu_handle = None
    training_start_time = time.time()

    try:
        # --- Setup ---
        general_config = config.get('general', {})
        seed = general_config.get('seed', 42)
        set_seed(seed)
        logger.info(f"Using random seed: {seed}")

        device_name = general_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_name)
        logger.info(f"Using device: {device}")

        # Initialize NVML for monitoring
        monitoring_config = config.get('monitoring', {})
        if device.type == 'cuda' and monitoring_config.get('enabled', True):
            if init_nvml():
                nvml_active = True
                gpu_index = device.index if device.index is not None else 0
                gpu_handle = get_gpu_handle(gpu_index)
                if gpu_handle:
                     logger.info(f"NVML initialized for GPU {gpu_index}.")
                     # Log initial memory usage
                     mem_info = get_gpu_memory_usage(gpu_handle)
                     if mem_info:
                          log_metrics({'initial_gpu_mem_used_mib': mem_info[0]}, wandb_run=wandb_run)
                else:
                     logger.warning(f"NVML initialized but failed to get handle for GPU {gpu_index}.")
                     nvml_active = False
            else:
                logger.warning("NVML initialization failed. GPU monitoring disabled.")

        # --- Data Loading ---
        data_config = config.get('dataset', {})
        loader_config = config.get('data_loader', {})
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_config.get('name', 'FashionMNIST'),
            batch_size=loader_config.get('batch_size', 64),
            data_root=data_config.get('root_dir', './data'),
            val_split=data_config.get('val_split', 0.1),
            seed=seed,
            num_workers=loader_config.get('num_workers', 4),
            pin_memory=loader_config.get('pin_memory', True) if device.type == 'cuda' else False,
            download=data_config.get('download', True)
        )
        logger.info("Dataloaders created.")

        # --- Model Instantiation ---
        model, input_adapter = get_model_and_adapter(config)
        model.to(device)
        logger.info(f"Model '{config.get('model', {}).get('name')}' instantiated and moved to {device}.")
        # Log model parameter count
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_metrics({'model_parameters': num_params}, wandb_run=wandb_run)
        logger.info(f"Model trainable parameters: {num_params:,}")


        # --- FLOPs Profiling ---
        profiling_config = config.get('profiling', {})
        if profiling_config.get('enabled', True):
            try:
                # Get a sample input shape from the loader
                sample_input, _ = next(iter(train_loader))
                if input_adapter: # Apply adapter if needed for profiling
                     sample_input = input_adapter(sample_input)
                input_shape = sample_input.shape # Use shape of potentially adapted input
                logger.info(f"Profiling FLOPs with input shape: {input_shape}")

                gmacs = profile_model_flops(
                    model,
                    input_shape=input_shape, # Pass shape tuple
                    device=device,
                    verbose=profiling_config.get('verbose', False)
                )
                if gmacs is not None:
                    results['estimated_gmacs'] = gmacs
                    # Estimate GFLOPs (often 2*MACs)
                    results['estimated_gflops'] = gmacs * 2.0
                    log_metrics({'estimated_gmacs': gmacs, 'estimated_gflops': gmacs * 2.0}, wandb_run=wandb_run)
                    logger.info(f"Estimated GFLOPs: {results['estimated_gflops']:.4f} G")
                else:
                    logger.warning("FLOPs profiling failed.")
            except Exception as e:
                logger.error(f"FLOPs profiling failed: {e}", exc_info=True)


        # --- Training ---
        logger.info("Starting training phase...")
        train_config = config.get('training', {})
        algorithm_name = config.get('algorithm', {}).get('name', '').lower()
        peak_gpu_mem = 0.0

        # TODO: Implement energy monitoring start/stop around training call
        # For now, just track peak memory during training loop (if possible within train functions)
        # Or sample periodically here? Simpler to log peak after training for now.

        train_loop_start_time = time.time()

        if algorithm_name == 'bp':
            train_bp_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config, # Pass the 'training' sub-config
                device=device,
                wandb_run=wandb_run,
                input_adapter=input_adapter
            )
        elif algorithm_name == 'cafo':
            train_cafo_model(
                model=model, # Should be CaFo_CNN instance
                train_loader=train_loader,
                config=train_config, # Pass the 'training' sub-config
                device=device,
                wandb_run=wandb_run
                # CaFo handles its own input adaptation internally
            )
        elif algorithm_name == 'mf':
             train_mf_model(
                 model=model, # Should be MF_MLP instance
                 train_loader=train_loader,
                 config=train_config, # Pass the 'training' sub-config
                 device=device,
                 wandb_run=wandb_run
                 # MF handles its own input adaptation internally
             )
        elif algorithm_name == 'ff':
             # train_ff_model(
             #     model=model, # Should be FF_MLP instance
             #     train_loader=train_loader,
             #     config=train_config,
             #     device=device,
             #     wandb_run=wandb_run
             # )
             raise NotImplementedError("train_ff_model orchestration not fully implemented yet.")
        else:
            raise ValueError(f"Unknown algorithm name for training: {algorithm_name}")

        train_loop_duration = time.time() - train_loop_start_time
        results['training_duration_sec'] = train_loop_duration
        log_metrics({'training_duration_sec': train_loop_duration}, wandb_run=wandb_run)
        logger.info(f"Training phase completed in: {format_time(train_loop_duration)}")

        # Log peak memory after training (if NVML active)
        if nvml_active and gpu_handle:
             mem_info = get_gpu_memory_usage(gpu_handle)
             if mem_info:
                  peak_gpu_mem = mem_info[0] # Used memory
                  results['peak_gpu_mem_used_mib'] = peak_gpu_mem
                  log_metrics({'peak_gpu_mem_used_mib': peak_gpu_mem}, wandb_run=wandb_run)
                  logger.info(f"Peak GPU Memory Usage (End of Training): {peak_gpu_mem:.2f} MiB")


        # --- Evaluation ---
        logger.info("Starting evaluation phase on test set...")
        eval_config = config.get('evaluation', {})
        # Use appropriate loss for evaluation (usually CrossEntropy for classification)
        eval_criterion_name = eval_config.get('criterion', 'CrossEntropyLoss')
        if eval_criterion_name.lower() == 'crossentropyloss':
            eval_criterion = nn.CrossEntropyLoss()
        else:
            # Allow evaluation without loss calculation
            logger.warning(f"Unsupported evaluation criterion: {eval_criterion_name}. Evaluating accuracy only.")
            eval_criterion = None

        eval_results = {}
        if algorithm_name == 'bp':
            eval_loss, eval_acc = evaluate_bp_model(
                model, test_loader, eval_criterion, device, input_adapter
            )
            eval_results = {'test_loss': eval_loss, 'test_accuracy': eval_acc}
        elif algorithm_name == 'cafo':
            # Evaluate using the last predictor by default
            eval_results = evaluate_cafo_model(
                model, test_loader, device, eval_criterion, use_predictor_index=-1
            )
            # Rename keys for consistency
            eval_results = {
                'test_loss': eval_results.get(f'eval_predictor_{len(model.predictors)}_loss', 0.0),
                'test_accuracy': eval_results.get(f'eval_predictor_{len(model.predictors)}_accuracy', 0.0)
            }
        elif algorithm_name == 'mf':
             eval_results = evaluate_mf_model(
                 model, test_loader, device, eval_criterion
             )
             # Rename keys
             eval_results = {
                 'test_loss': eval_results.get('eval_loss', 0.0),
                 'test_accuracy': eval_results.get('eval_accuracy', 0.0)
             }
        elif algorithm_name == 'ff':
             # eval_results = evaluate_ff_model(model, test_loader, device) # Needs implementation
             # eval_results = {'test_accuracy': eval_results.get('eval_accuracy', 0.0)}
             raise NotImplementedError("evaluate_ff_model not implemented yet.")
        else:
             raise ValueError(f"Unknown algorithm name for evaluation: {algorithm_name}")


        logger.info(f"Test Set Evaluation: Accuracy: {eval_results.get('test_accuracy', 0.0):.2f}%, Loss: {eval_results.get('test_loss', 0.0):.4f}")
        results.update(eval_results)
        log_metrics(eval_results, wandb_run=wandb_run) # Log final test metrics


    except Exception as e:
        logger.error(f"An error occurred during the training/evaluation run: {e}", exc_info=True)
        results['error'] = str(e)
        # Optionally re-raise the exception
        # raise e
    finally:
        # --- Cleanup ---
        if nvml_active:
            shutdown_nvml()
            logger.info("NVML shut down.")

        total_run_time = time.time() - training_start_time
        results['total_duration_sec'] = total_run_time
        log_metrics({'total_duration_sec': total_run_time}, wandb_run=wandb_run)
        logger.info(f"Total run duration: {format_time(total_run_time)}")

        # Finish W&B run if it was initialized here (usually done in main script)
        # if wandb_run:
        #     wandb_run.finish()

    return results


if __name__ == '__main__':
    # Example usage: Load a config file and run training
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Testing Training Engine...")

    # Create dummy config files for testing
    # Base config
    base_config_content = """
general:
  seed: 123
  device: cpu # Use CPU for basic test unless GPU is known available
dataset:
  name: FashionMNIST
  root_dir: ./data_test_engine
  num_classes: 10
  input_channels: 1
  image_size: 28
data_loader:
  batch_size: 8 # Small batch for quick test
  num_workers: 0
monitoring:
  enabled: false # Disable GPU monitoring for CPU test
profiling:
  enabled: true
training:
  optimizer: Adam
  optimizer_params:
    lr: 0.01
  log_interval: 2
evaluation:
  criterion: CrossEntropyLoss
"""
    # BP config
    bp_config_content = """
algorithm:
  name: BP
model:
  name: MF_MLP # Use MF_MLP structure for BP test
  params:
    hidden_dims: [64, 32] # Smaller MLP for testing
    activation: relu
training:
  epochs: 1 # Just one epoch
  criterion: CrossEntropyLoss
"""
    # CaFo config
    cafo_config_content = """
algorithm:
  name: CaFo
model:
  name: CaFo_CNN
  params:
    block_channels: [8, 16] # Smaller CNN for testing
training:
  epochs_per_block: 1
  criterion: CrossEntropyLoss
"""

    os.makedirs('configs_test_engine', exist_ok=True)
    with open('configs_test_engine/base_test.yaml', 'w') as f:
        f.write(base_config_content)
    with open('configs_test_engine/bp_test.yaml', 'w') as f:
        f.write(bp_config_content)
    with open('configs_test_engine/cafo_test.yaml', 'w') as f:
        f.write(cafo_config_content)

    # --- Test BP Run ---
    print("\n--- Testing BP Run ---")
    try:
        config_bp = load_config('configs_test_engine/bp_test.yaml', base_config_path='configs_test_engine/base_test.yaml')
        # Set WANDB_MODE=offline if testing wandb without login
        # os.environ['WANDB_MODE'] = 'offline'
        # wandb_run_bp = setup_wandb(config_bp, project_name="Test-Engine-BP", run_name="bp_engine_test")
        results_bp = run_training(config_bp, wandb_run=None) # Test without wandb first
        print("BP Run Results:", results_bp)
        assert 'error' not in results_bp
        assert 'test_accuracy' in results_bp
    except Exception as e:
        print(f"BP run failed: {e}", exc_info=True)

    # --- Test CaFo Run ---
    print("\n--- Testing CaFo Run ---")
    try:
        config_cafo = load_config('configs_test_engine/cafo_test.yaml', base_config_path='configs_test_engine/base_test.yaml')
        # wandb_run_cafo = setup_wandb(config_cafo, project_name="Test-Engine-CaFo", run_name="cafo_engine_test")
        results_cafo = run_training(config_cafo, wandb_run=None)
        print("CaFo Run Results:", results_cafo)
        assert 'error' not in results_cafo
        assert 'test_accuracy' in results_cafo
    except Exception as e:
        print(f"CaFo run failed: {e}", exc_info=True)

    # Clean up dummy configs and data
    import shutil
    # shutil.rmtree('configs_test_engine')
    # if os.path.exists('./data_test_engine'):
    #      shutil.rmtree('./data_test_engine')
    print("\nEngine testing finished. Manual cleanup of 'configs_test_engine' and 'data_test_engine' might be needed.")
