import pynvml
import time
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

_nvml_initialized = False
_gpu_handles: Dict[int, pynvml.c_nvmlDevice_t] = {}

def init_nvml():
    """Initializes the NVML library."""
    global _nvml_initialized
    if _nvml_initialized:
        logger.debug("NVML already initialized.")
        return True
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
        logger.info("NVML initialized successfully.")
        # Log driver and NVML versions
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        nvml_version = pynvml.nvmlSystemGetNvmlVersion()
        logger.info(f"NVIDIA Driver Version: {driver_version}")
        logger.info(f"NVML Library Version: {nvml_version}")
        return True
    except pynvml.NVMLError as error:
        logger.error(f"Failed to initialize NVML: {error}", exc_info=True)
        _nvml_initialized = False
        return False

def shutdown_nvml():
    """Shuts down the NVML library."""
    global _nvml_initialized, _gpu_handles
    if not _nvml_initialized:
        logger.debug("NVML not initialized, skipping shutdown.")
        return
    try:
        pynvml.nvmlShutdown()
        _nvml_initialized = False
        _gpu_handles = {} # Clear cached handles
        logger.info("NVML shut down successfully.")
    except pynvml.NVMLError as error:
        logger.error(f"Failed to shut down NVML: {error}", exc_info=True)

def get_gpu_handle(device_index: int = 0) -> Optional[pynvml.c_nvmlDevice_t]:
    """
    Gets the NVML handle for a specific GPU device.

    Args:
        device_index: The index of the GPU device (usually 0 for single-GPU setups).

    Returns:
        The NVML device handle, or None if NVML is not initialized or the device is not found.
    """
    if not _nvml_initialized:
        logger.error("NVML not initialized. Cannot get GPU handle.")
        if not init_nvml(): # Try to initialize automatically
             return None

    if device_index in _gpu_handles:
        return _gpu_handles[device_index]

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        _gpu_handles[device_index] = handle
        logger.debug(f"Retrieved handle for GPU device {device_index}.")
        return handle
    except pynvml.NVMLError as error:
        logger.error(f"Failed to get handle for GPU device {device_index}: {error}", exc_info=True)
        return None

def get_gpu_power_usage(handle: pynvml.c_nvmlDevice_t) -> Optional[float]:
    """
    Gets the current power usage of the GPU in Watts.

    Args:
        handle: The NVML device handle.

    Returns:
        Power usage in Watts, or None if an error occurs.
    """
    if not handle:
        logger.error("Invalid GPU handle provided for power usage query.")
        return None
    try:
        # Power usage is reported in milliwatts
        power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_watts = power_milliwatts / 1000.0
        return power_watts
    except pynvml.NVMLError as error:
        # NVMLError_NotSupported can happen on some systems/GPUs
        if error.value == pynvml.NVMLError_NotSupported:
             logger.warning(f"Power usage reporting not supported for this GPU.")
        else:
             logger.error(f"Failed to get power usage: {error}", exc_info=True)
        return None

def get_gpu_memory_usage(handle: pynvml.c_nvmlDevice_t) -> Optional[Tuple[float, float, float]]:
    """
    Gets the memory usage of the GPU in MiB.

    Args:
        handle: The NVML device handle.

    Returns:
        A tuple containing (used_memory_mib, total_memory_mib, free_memory_mib),
        or None if an error occurs.
    """
    if not handle:
        logger.error("Invalid GPU handle provided for memory usage query.")
        return None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Convert bytes to MiB (1 MiB = 1024 * 1024 bytes)
        bytes_to_mib = 1 / (1024 * 1024)
        total_mib = mem_info.total * bytes_to_mib
        used_mib = mem_info.used * bytes_to_mib
        free_mib = mem_info.free * bytes_to_mib # Or calculate as total_mib - used_mib
        return used_mib, total_mib, free_mib
    except pynvml.NVMLError as error:
        logger.error(f"Failed to get memory usage: {error}", exc_info=True)
        return None

# --- Placeholder for Energy Monitoring ---
# This requires a more complex implementation, likely involving periodic sampling
# in a separate thread or integrated directly into the training loop's steps.

# Example:
# _monitoring_active = False
# _energy_samples = []
# _monitoring_thread = None
# _monitoring_interval = 0.1 # seconds

# def _energy_monitor_thread(handle, interval):
#     global _monitoring_active, _energy_samples
#     while _monitoring_active:
#         power = get_gpu_power_usage(handle)
#         if power is not None:
#             _energy_samples.append((time.time(), power))
#         time.sleep(interval)

# def start_energy_monitoring(handle, interval_sec=0.1):
#     # ... implementation using threading.Thread ...
#     pass

# def stop_energy_monitoring():
#     # ... stop thread, calculate integral (trapezoidal rule) ...
#     # Energy (Joules) = sum( (power_i + power_{i+1})/2 * (time_{i+1} - time_i) )
#     pass
# --- End Placeholder ---


if __name__ == '__main__':
    print("Testing monitoring_utils...")

    if init_nvml():
        print("NVML Initialized.")
        num_devices = pynvml.nvmlDeviceGetCount()
        print(f"Found {num_devices} GPU device(s).")

        if num_devices > 0:
            handle = get_gpu_handle(0)
            if handle:
                print("\n--- GPU 0 ---")
                device_name = pynvml.nvmlDeviceGetName(handle)
                print(f"Device Name: {device_name}")

                power = get_gpu_power_usage(handle)
                if power is not None:
                    print(f"Current Power Usage: {power:.2f} W")
                else:
                    print("Could not retrieve power usage.")

                memory = get_gpu_memory_usage(handle)
                if memory is not None:
                    used, total, free = memory
                    print(f"Memory Usage: {used:.2f} MiB Used / {total:.2f} MiB Total ({free:.2f} MiB Free)")
                else:
                    print("Could not retrieve memory usage.")

                # Test getting handle again (should use cache)
                handle_cached = get_gpu_handle(0)
                assert handle_cached == handle
                print("\nHandle caching test passed.")

            else:
                print("Could not get handle for GPU 0.")
        else:
            print("No compatible NVIDIA GPU found.")

        shutdown_nvml()
        print("\nNVML Shut Down.")
    else:
        print("NVML initialization failed. Cannot perform tests.")
