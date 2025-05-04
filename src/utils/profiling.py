# --------------------------------------------------------------------------------
# File: ./src/utils/profiling.py
# --------------------------------------------------------------------------------
# File: src/utils/profiling.py
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from typing import Tuple, Optional, Union, Callable

logger = logging.getLogger(__name__)

def profile_model_flops(
    model: nn.Module,
    input_constructor: Union[Tuple[int, ...], Callable[[], torch.Tensor]],
    device: Optional[torch.device] = None,
    verbose: bool = False, # Verbose now controls printing the profiler table
) -> Optional[float]:
    """
    Estimates the FLOPs for a model's forward pass using torch.profiler.

    Args:
        model: The PyTorch model (nn.Module) to profile.
        input_constructor: A tuple describing the input shape (e.g., (1, 3, 32, 32)) for random input,
                           OR a callable function that returns a sample input tensor.
                           The callable should return a tensor suitable for the model's *forward* method
                           (e.g., already flattened if needed). Batch size should typically be 1.
        device: The device ('cuda' or 'cpu') to run the profiling on. If None, uses model's device or CPU default.
        verbose: If True, prints the detailed profiler summary sorted by CPU time.

    Returns:
        Estimated GigaFLOPs (GFLOPs), or None if profiling fails. Returns 0.0 if no FLOPs are recorded.
    """
    if device is None:
        try: device = next(model.parameters()).device
        except StopIteration: device = torch.device("cpu"); logger.debug("Model has no params, using CPU for profiling.")
        except Exception as e: logger.warning(f"Could not infer model device, using CPU: {e}"); device = torch.device("cpu")
    logger.debug(f"Profiling with torch.profiler on device: {device}")

    original_mode = model.training
    model.eval()
    model.to(device)

    try:
        if isinstance(input_constructor, tuple):
            input_shape = input_constructor
            if input_shape[0] != 1:
                logger.warning(f"Input shape tuple has batch size {input_shape[0]}, changing to 1 for FLOPs profiling.")
                input_shape = (1,) + input_shape[1:]
            dummy_input = torch.randn(input_shape, device=device)
            logger.debug(f"Created dummy input with shape: {input_shape}")
        elif callable(input_constructor):
            dummy_input = input_constructor()
            dummy_input = dummy_input.to(device) # Ensure input is on the correct device
            if dummy_input.shape[0] != 1:
                 logger.warning(f"Input constructor yielded batch size {dummy_input.shape[0]}, expected 1 for profiling. Using first sample.")
                 dummy_input = dummy_input[:1] # Use only the first sample
            logger.debug(f"Created dummy input from constructor, shape: {dummy_input.shape}")
        else:
            logger.error("Invalid input_constructor type provided.")
            model.train(original_mode); return None
    except Exception as e:
        logger.error(f"Failed to create dummy input: {e}", exc_info=True)
        model.train(original_mode); return None

    total_flops = 0.0 # Initialize total FLOPs
    try:
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=False, profile_memory=False, with_flops=True) as prof:
             with record_function("model_forward_pass"): # Label the operation
                 with torch.no_grad():
                     _ = model(dummy_input) # Run forward pass

        if verbose:
            print("--- Torch Profiler Results (Sorted by CPU time) ---")
            try:
                # Sort by flops if possible, otherwise fallback to CPU time
                sort_key = "flops" if any(e.flops > 0 for e in prof.key_averages()) else "self_cpu_time_total"
                print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
            except Exception as table_err:
                 logger.warning(f"Could not generate profiler table: {table_err}")
            print("--------------------------------------------------")


        # Sum FLOPs from recorded events
        for event in prof.key_averages():
             # event.flops seems to report total FLOPs for the operator
             if event.flops > 0:
                 total_flops += event.flops

        if total_flops == 0:
             logger.warning("torch.profiler recorded 0 FLOPs. Check model or input.")

    except Exception as e:
        logger.error(f"Failed during torch.profiler execution: {e}", exc_info=True)
        total_flops = None
    finally:
        model.train(original_mode) # Restore original mode

    if total_flops is not None:
        gflops = total_flops / 1e9 # Calculate GigaFLOPs
        # Log GFLOPs instead of MACs
        logger.info(f"Estimated Total Forward Pass FLOPs (via torch.profiler): {gflops:.4f} GFLOPs")
        return gflops # Return GFLOPs
    else:
        logger.warning("Failed to determine Total FLOPs from torch.profiler.")
        return None