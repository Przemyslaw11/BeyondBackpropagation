import torch
import torch.nn as nn
import torchprof
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

def profile_model_flops(model: nn.Module, input_shape: Union[Tuple[int, ...], torch.Tensor], device: Optional[torch.device] = None, verbose: bool = False) -> Optional[float]:
    """
    Estimates the FLOPs (specifically MACs reported by torchprof) for a model's forward pass.

    Args:
        model: The PyTorch model (nn.Module) to profile.
        input_shape: A tuple describing the input shape (e.g., (1, 3, 32, 32) for a single image)
                     or a sample input tensor. Batch size should typically be 1 for representative FLOPs.
        device: The device ('cuda' or 'cpu') to run the profiling on. If None, uses model's device or default.
        verbose: If True, prints the detailed torchprof summary.

    Returns:
        Estimated GFLOPs (Giga Floating Point Operations, often MACs * 2),
        or None if profiling fails.
    """
    if device is None:
        # Try to infer device from model parameters, default to CPU
        try:
            device = next(model.parameters()).device
        except StopIteration: # Model might have no parameters
             device = torch.device("cpu")
             logger.debug("Model has no parameters, defaulting device to CPU for profiling.")
        except Exception as e:
             logger.warning(f"Could not infer model device, defaulting to CPU: {e}")
             device = torch.device("cpu")


    # Ensure model is on the correct device and in eval mode
    original_mode = model.training
    model.eval()
    model.to(device)

    # Create dummy input tensor if shape is provided
    if isinstance(input_shape, tuple):
        try:
            dummy_input = torch.randn(input_shape, device=device)
        except Exception as e:
            logger.error(f"Failed to create dummy input tensor with shape {input_shape} on device {device}: {e}", exc_info=True)
            return None
    elif isinstance(input_shape, torch.Tensor):
        dummy_input = input_shape.to(device)
    else:
        logger.error("Invalid input_shape provided. Must be a tuple or a torch.Tensor.")
        return None

    total_macs = None
    try:
        # Profile the forward pass
        with torchprof.Profile(model, use_cuda=(device.type == 'cuda'), profile_memory=False) as prof: # Memory profiling can be slow
            with torch.no_grad():
                model(dummy_input)

        if verbose:
            # Print the detailed summary (includes MACs per layer)
            print(prof)

        # Extract total MACs (Multiply-Accumulate operations)
        # torchprof reports MACs. Often, FLOPs are estimated as 2 * MACs.
        # We will return GigaMACs and let the user decide how to interpret as FLOPs.
        # The summary string often contains 'Total MACs:'
        summary_str = str(prof) # Get the string representation
        for line in summary_str.split('\n'):
             if 'Total MACs:' in line:
                  mac_str = line.split(':')[-1].strip()
                  # Handle suffixes like K, M, G
                  if mac_str.endswith('K'):
                      total_macs = float(mac_str[:-1]) * 1e3
                  elif mac_str.endswith('M'):
                      total_macs = float(mac_str[:-1]) * 1e6
                  elif mac_str.endswith('G'):
                      total_macs = float(mac_str[:-1]) * 1e9
                  else:
                      total_macs = float(mac_str)
                  break

        if total_macs is None:
             logger.warning("Could not parse Total MACs from torchprof summary.")
             # Fallback: try accessing trace events if available (might be fragile)
             try:
                 total_macs = prof.total_macs() # Check if a direct method exists (may vary by version)
             except AttributeError:
                 logger.warning("torchprof object does not have a 'total_macs' method.")


    except Exception as e:
        logger.error(f"Failed to profile model FLOPs: {e}", exc_info=True)
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


if __name__ == '__main__':
    print("Testing profiling_utils...")

    # Define simple models
    linear_model = nn.Linear(10, 5)
    conv_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 16 * 16, 10) # Assuming 32x32 input -> 16x16 after pool
    )

    # Test on CPU
    print("\n--- Profiling Linear Model (CPU) ---")
    input_shape_linear = (1, 10) # Batch size 1
    gmacs_linear_cpu = profile_model_flops(linear_model, input_shape_linear, device=torch.device('cpu'), verbose=True)
    if gmacs_linear_cpu is not None:
        print(f"Estimated GigaMACs (CPU): {gmacs_linear_cpu:.6f}")
        # Expected MACs for Linear(in, out) = batch_size * in_features * out_features = 1 * 10 * 5 = 50
        # Expected GigaMACs = 50 / 1e9 = 0.00000005
        assert abs(gmacs_linear_cpu * 1e9 - 50) < 1e-9 # Check if close to 50 MACs

    print("\n--- Profiling Conv Model (CPU) ---")
    input_shape_conv = (1, 3, 32, 32) # Batch size 1
    gmacs_conv_cpu = profile_model_flops(conv_model, input_shape_conv, device=torch.device('cpu'), verbose=True)
    if gmacs_conv_cpu is not None:
        print(f"Estimated GigaMACs (CPU): {gmacs_conv_cpu:.6f}")
        # Manual calculation is more complex, but we expect a non-zero value.

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n--- Profiling Linear Model (GPU) ---")
        gmacs_linear_gpu = profile_model_flops(linear_model, input_shape_linear, device=torch.device('cuda'), verbose=False) # Less verbose for GPU
        if gmacs_linear_gpu is not None:
            print(f"Estimated GigaMACs (GPU): {gmacs_linear_gpu:.6f}")
            assert abs(gmacs_linear_gpu * 1e9 - 50) < 1e-9 # Should be the same MAC count

        print("\n--- Profiling Conv Model (GPU) ---")
        gmacs_conv_gpu = profile_model_flops(conv_model, input_shape_conv, device=torch.device('cuda'), verbose=False)
        if gmacs_conv_gpu is not None:
            print(f"Estimated GigaMACs (GPU): {gmacs_conv_gpu:.6f}")
            # Compare CPU and GPU results (should be very close)
            if gmacs_conv_cpu is not None:
                 print(f"Difference CPU vs GPU GigaMACs: {abs(gmacs_conv_cpu - gmacs_conv_gpu):.6f}")
                 assert abs(gmacs_conv_cpu - gmacs_conv_gpu) < 1e-5 # Allow small tolerance
    else:
        print("\nCUDA not available, skipping GPU profiling tests.")

    # Test with tensor input
    print("\n--- Profiling with Tensor Input (CPU) ---")
    dummy_tensor = torch.randn(input_shape_linear)
    gmacs_tensor_input = profile_model_flops(linear_model, dummy_tensor, device=torch.device('cpu'), verbose=False)
    if gmacs_tensor_input is not None:
        print(f"Estimated GigaMACs (Tensor Input): {gmacs_tensor_input:.6f}")
        assert abs(gmacs_tensor_input * 1e9 - 50) < 1e-9
