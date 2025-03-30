# File: src/utils/profiling.py
import torch
import torch.nn as nn
import torchprof
import logging
from typing import Tuple, Optional, Union, Callable  # Added Callable

logger = logging.getLogger(__name__)


def profile_model_flops(
    model: nn.Module,
    input_constructor: Union[
        Tuple[int, ...], Callable[[], torch.Tensor]
    ],  # Shape tuple or function returning input tensor
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Optional[float]:
    """
    Estimates the FLOPs (specifically MACs reported by torchprof) for a model's forward pass.

    Args:
        model: The PyTorch model (nn.Module) to profile.
        input_constructor: A tuple describing the input shape (e.g., (1, 3, 32, 32)) for random input,
                           OR a callable function that returns a sample input tensor.
                           Batch size should typically be 1 for representative FLOPs.
        device: The device ('cuda' or 'cpu') to run the profiling on. If None, uses model's device or CPU default.
        verbose: If True, prints the detailed torchprof summary.

    Returns:
        Estimated GigaMACs (Giga Multiply-Accumulate operations), or None if profiling fails.
        Note: GFLOPs is often estimated as 2 * GigaMACs.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # Model might have no parameters
            device = torch.device("cpu")
            logger.debug(
                "Model has no parameters, defaulting device to CPU for profiling."
            )
        except Exception as e:
            logger.warning(f"Could not infer model device, defaulting to CPU: {e}")
            device = torch.device("cpu")
    logger.debug(f"Profiling on device: {device}")

    # Ensure model is on the correct device and in eval mode
    original_mode = model.training
    try:
        model.eval()
        model.to(device)

        # Create dummy input tensor
        if isinstance(input_constructor, tuple):
            input_shape = input_constructor
            try:
                dummy_input = torch.randn(input_shape, device=device)
                logger.debug(f"Created dummy input with shape: {input_shape}")
            except Exception as e:
                logger.error(
                    f"Failed to create dummy input tensor with shape {input_shape} on device {device}: {e}",
                    exc_info=True,
                )
                return None
        elif callable(input_constructor):
            try:
                dummy_input = input_constructor().to(device)
                logger.debug(
                    f"Created dummy input from constructor, shape: {dummy_input.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create dummy input using constructor function: {e}",
                    exc_info=True,
                )
                return None
        else:
            logger.error(
                "Invalid input_constructor provided. Must be a shape tuple or a callable returning a tensor."
            )
            return None

        total_macs = None
        # --- Profile the forward pass ---
        with torchprof.Profile(
            model, use_cuda=(device.type == "cuda"), profile_memory=False
        ) as prof:  # Memory profiling can be slow and less relevant here
            with torch.no_grad():
                model(dummy_input)

        if verbose:
            try:
                print(prof)  # Print the default summary
                # Or for more control: print(prof.display(show_events=False))
            except Exception as e:
                logger.warning(f"Could not print torchprof summary: {e}")

        # --- Extract total MACs ---
        # Attempt 1: Use the string representation (common method)
        try:
            summary_str = str(prof)
            for line in summary_str.split("\n"):
                if "Total MACs:" in line:
                    mac_str = (
                        line.split(":")[-1].strip().upper()
                    )  # Handle case-insensitivity
                    value = float(mac_str[:-1])  # Remove suffix
                    if mac_str.endswith("K"):
                        total_macs = value * 1e3
                    elif mac_str.endswith("M"):
                        total_macs = value * 1e6
                    elif mac_str.endswith("G"):
                        total_macs = value * 1e9
                    elif mac_str.endswith("T"):
                        total_macs = value * 1e12
                    else:  # Assume no suffix or unrecognized suffix
                        total_macs = float(mac_str)  # Try converting directly
                    logger.debug(f"Parsed MACs from summary string: {total_macs}")
                    break
        except Exception as e:
            logger.warning(
                f"Could not parse Total MACs from torchprof summary string: {e}"
            )
            total_macs = None  # Ensure reset if parsing fails

        # Attempt 2: Use internal methods if available (may vary by torchprof version)
        if total_macs is None:
            try:
                # Check for common attribute names used by profilers
                if hasattr(prof, "total_macs"):
                    total_macs = prof.total_macs()
                    logger.debug(f"Retrieved MACs from prof.total_macs(): {total_macs}")
                # Add other potential attribute names if needed
            except Exception as e:
                logger.warning(f"Could not retrieve MACs using internal methods: {e}")

        if total_macs is None:
            logger.warning("Failed to determine Total MACs from torchprof.")

    except ImportError:
        logger.error(
            "torchprof not found. Please install it (`pip install torchprof`) for FLOPs profiling."
        )
        total_macs = None
    except Exception as e:
        logger.error(f"Failed to profile model MACs/FLOPs: {e}", exc_info=True)
        total_macs = None
    finally:
        # Restore original training mode
        model.train(original_mode)

    if total_macs is not None:
        gmacs = total_macs / 1e9
        logger.info(f"Estimated Total MACs: {gmacs:.4f} G")
        # Return GigaMACs. User can multiply by 2 for GFLOPs estimate if desired.
        return gmacs
    else:
        return None
