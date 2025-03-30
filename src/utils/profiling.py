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
        # Wrap model forward if it requires specific args (unlikely for standard models)
        forward_args = (dummy_input,)
        # Note: For models like CaFo needing specific forward methods, adjust here or profile externally.
        # This default profiles the __call__ method which usually calls forward().

        with torchprof.Profile(
            model, use_cuda=(device.type == "cuda"), profile_memory=False
        ) as prof:
            with torch.no_grad():
                # Call the model directly
                model(*forward_args)

        if verbose:
            try:
                print(prof)
            except Exception as e:
                logger.warning(f"Could not print torchprof summary: {e}")

        # --- Extract total MACs ---
        try:
            # Using the internal structured data is more robust if available
            ops = prof.raw()  # Get raw operator data
            total_macs = 0
            for op in ops:
                if (
                    hasattr(op, "macs") and op.macs
                ):  # Check if macs attribute exists and is not None/0
                    total_macs += op.macs
            if (
                total_macs == 0
            ):  # Fallback to parsing string if internal method yields 0 or fails
                raise ValueError("Internal MACs sum is zero, trying string parsing.")
            logger.debug(f"Retrieved MACs from prof.raw(): {total_macs}")
        except Exception as e_internal:
            logger.debug(
                f"Could not retrieve MACs using internal method ({e_internal}), trying string parsing."
            )
            total_macs = None  # Reset for string parsing
            try:
                summary_str = str(prof)
                for line in summary_str.split("\n"):
                    if "Total MACs:" in line:
                        mac_str = line.split(":")[-1].strip().upper()
                        value = float(mac_str[:-1])
                        if mac_str.endswith("K"):
                            total_macs = value * 1e3
                        elif mac_str.endswith("M"):
                            total_macs = value * 1e6
                        elif mac_str.endswith("G"):
                            total_macs = value * 1e9
                        elif mac_str.endswith("T"):
                            total_macs = value * 1e12
                        else:
                            total_macs = float(mac_str)
                        logger.debug(f"Parsed MACs from summary string: {total_macs}")
                        break
            except Exception as e_str:
                logger.warning(
                    f"Could not parse Total MACs from torchprof summary string: {e_str}"
                )
                total_macs = None

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
        model.train(original_mode)  # Restore original training mode

    if total_macs is not None and total_macs > 0:
        gmacs = total_macs / 1e9
        logger.info(f"Estimated Total MACs: {gmacs:.4f} G")
        return gmacs
    else:
        if total_macs == 0:
            logger.warning(
                "FLOPs profiling resulted in 0 MACs. Check model structure or torchprof compatibility."
            )
        return None
