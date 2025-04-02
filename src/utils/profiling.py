# File: src/utils/profiling.py (REVISED to use torch.profiler)
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from typing import Tuple, Optional, Union, Callable

logger = logging.getLogger(__name__)

def profile_model_flops(
    model: nn.Module,
    input_constructor: Union[
        Tuple[int, ...], Callable[[], torch.Tensor]
    ],
    device: Optional[torch.device] = None,
    verbose: bool = False, # Verbose now controls printing the profiler table
) -> Optional[float]:
    """
    Estimates the FLOPs (specifically MACs using torch.profiler) for a model's forward pass.

    Args:
        model: The PyTorch model (nn.Module) to profile.
        input_constructor: A tuple describing the input shape (e.g., (1, 3, 32, 32)) for random input,
                           OR a callable function that returns a sample input tensor.
                           Batch size should typically be 1 for representative FLOPs.
        device: The device ('cuda' or 'cpu') to run the profiling on. If None, uses model's device or CPU default.
        verbose: If True, prints the detailed profiler summary sorted by CPU time.

    Returns:
        Estimated GigaMACs (Giga Multiply-Accumulate operations), or None if profiling fails.
        Note: GFLOPs is often estimated as 2 * GigaMACs. Returns 0.0 if no FLOPs are recorded.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
            logger.debug("Model has no parameters, defaulting device to CPU for profiling.")
        except Exception as e:
            logger.warning(f"Could not infer model device, defaulting to CPU: {e}")
            device = torch.device("cpu")
    logger.debug(f"Profiling with torch.profiler on device: {device}")

    # Ensure model is on the correct device and in eval mode
    original_mode = model.training
    model.eval()
    model.to(device)

    # Create dummy input tensor
    try:
        if isinstance(input_constructor, tuple):
            input_shape = input_constructor
            dummy_input = torch.randn(input_shape, device=device)
            logger.debug(f"Created dummy input with shape: {input_shape}")
        elif callable(input_constructor):
            dummy_input = input_constructor().to(device)
            logger.debug(f"Created dummy input from constructor, shape: {dummy_input.shape}")
        else:
            logger.error("Invalid input_constructor provided.")
            model.train(original_mode) # Restore mode
            return None
    except Exception as e:
        logger.error(f"Failed to create dummy input: {e}", exc_info=True)
        model.train(original_mode) # Restore mode
        return None

    # Profile the forward pass using torch.profiler
    total_macs = 0.0
    try:
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=False, profile_memory=False, with_flops=True) as prof:
             with record_function("model_inference"): # Add a label
                 with torch.no_grad():
                     model(dummy_input) # Call the model

        if verbose:
            # Print sorted by CPU total time
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        # Extract FLOPs (reported as MACs by PyTorch profiler's flop_count_table)
        # Summing 'flops' from key_averages seems the most straightforward way
        # Note: PyTorch profiler often counts MACs and labels them as 'flops'.
        # We will treat the reported 'flops' as MACs here.
        for event in prof.key_averages():
             # event.flops is the total count for that operator instance
             if event.flops > 0: # event.flops should be the MAC count
                 total_macs += event.flops

        if total_macs == 0:
             logger.warning("torch.profiler recorded 0 FLOPs/MACs. Check model or profiler setup.")


    except Exception as e:
        logger.error(f"Failed during torch.profiler execution: {e}", exc_info=True)
        total_macs = None # Indicate failure
    finally:
        model.train(original_mode)  # Restore original training mode

    if total_macs is not None:
        gmacs = total_macs / 1e9
        logger.info(f"Estimated Total MACs (via torch.profiler): {gmacs:.4f} G")
        return gmacs
    else:
        logger.warning("Failed to determine Total MACs from torch.profiler.")
        return None