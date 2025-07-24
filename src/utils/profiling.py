"""Model profiling utilities, primarily for FLOPs estimation."""

import logging
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

logger = logging.getLogger(__name__)


def _get_profiling_device(
    model: nn.Module, device: Optional[torch.device]
) -> torch.device:
    """Determines the device to use for profiling."""
    if device:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        logger.debug("Model has no params, using CPU for profiling.")
        return torch.device("cpu")
    except Exception as e:
        logger.warning(f"Could not infer model device, using CPU: {e}")
        return torch.device("cpu")


def _create_profiling_input(
    input_constructor: Union[Tuple[int, ...], Callable[[], torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """Creates a dummy input tensor for profiling."""
    if isinstance(input_constructor, tuple):
        input_shape = input_constructor
        if input_shape[0] != 1:
            logger.warning(
                f"Input shape tuple has batch size {input_shape[0]}, changing to 1 for FLOPs profiling."
            )
            input_shape = (1,) + input_shape[1:]
        dummy_input = torch.randn(input_shape, device=device)
        logger.debug(f"Created dummy input with shape: {dummy_input.shape}")
    elif callable(input_constructor):
        dummy_input = input_constructor()
        dummy_input = dummy_input.to(device)
        if dummy_input.shape[0] != 1:
            logger.warning(
                f"Input constructor yielded batch size {dummy_input.shape[0]}, "
                "expected 1 for profiling. Using first sample."
            )
            dummy_input = dummy_input[:1]
        logger.debug(
            f"Created dummy input from constructor, shape: {dummy_input.shape}"
        )
    else:
        raise TypeError("Invalid input_constructor type provided.")
    return dummy_input


def profile_model_flops(
    model: nn.Module,
    input_constructor: Union[Tuple[int, ...], Callable[[], torch.Tensor]],
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Optional[float]:
    """Estimates the FLOPs for a model's forward pass using torch.profiler.

    Args:
        model: The PyTorch model (nn.Module) to profile.
        input_constructor: A tuple describing the input shape
            (e.g., (1, 3, 32, 32)) for random input, OR a callable
            function that returns a sample input tensor. The callable
            should return a tensor suitable for the model's *forward*
            method (e.g., already flattened). Batch size should be 1.
        device: The device ('cuda' or 'cpu') to run the profiling on. If None,
            uses the model's device or defaults to CPU.
        verbose: If True, prints the detailed profiler summary sorted by CPU time.

    Returns:
        Estimated GigaFLOPs (GFLOPs), or None if profiling fails.
        Returns 0.0 if no FLOPs are recorded.
    """
    resolved_device = _get_profiling_device(model, device)
    logger.debug(f"Profiling with torch.profiler on device: {resolved_device}")

    original_mode = model.training
    model.eval()
    model.to(resolved_device)

    try:
        dummy_input = _create_profiling_input(input_constructor, resolved_device)
    except (TypeError, Exception) as e:
        logger.error(f"Failed to create dummy input: {e}", exc_info=True)
        model.train(original_mode)
        return None

    total_flops = 0.0
    try:
        activities = [ProfilerActivity.CPU]
        if resolved_device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with (
            profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof,
            record_function("model_forward_pass"),
            torch.no_grad(),
        ):
            _ = model(dummy_input)

        if verbose:
            print("--- Torch Profiler Results (Sorted by CPU time) ---")
            try:
                has_flops = any(e.flops > 0 for e in prof.key_averages())
                sort_key = "flops" if has_flops else "self_cpu_time_total"
                print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
            except Exception as table_err:
                logger.warning(f"Could not generate profiler table: {table_err}")
            print("--------------------------------------------------")

        for event in prof.key_averages():
            total_flops += event.flops or 0

        if total_flops == 0:
            logger.warning("torch.profiler recorded 0 FLOPs. Check model or input.")

    except Exception as e:
        logger.error(f"Failed during torch.profiler execution: {e}", exc_info=True)
        total_flops = None
    finally:
        model.train(original_mode)

    if total_flops is not None:
        gflops = total_flops / 1e9
        logger.info(
            f"Estimated Total Forward Pass FLOPs (via torch.profiler): {gflops:.4f} GFLOPs"
        )
        return gflops
    else:
        logger.warning("Failed to determine Total FLOPs from torch.profiler.")
        return None
