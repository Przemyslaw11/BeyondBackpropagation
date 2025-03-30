# File: src/utils/monitoring.py
import torch
import pynvml
import time
import logging
import threading
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger(__name__)

_nvml_initialized = False
_nvml_lock = threading.Lock()  # Lock for thread-safe init/shutdown
_gpu_handles: Dict[int, pynvml.c_nvmlDevice_t] = {}


def init_nvml():
    """Initializes the NVML library (thread-safe)."""
    global _nvml_initialized
    with _nvml_lock:
        if _nvml_initialized:
            logger.debug("NVML already initialized.")
            return True
        try:
            pynvml.nvmlInit()
            _nvml_initialized = True
            logger.info("NVML initialized successfully.")
            # Log driver and NVML versions only once
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                nvml_version = pynvml.nvmlSystemGetNvmlVersion()
                logger.info(f"NVIDIA Driver Version: {driver_version}")
                logger.info(f"NVML Library Version: {nvml_version}")
            except pynvml.NVMLError as e:
                logger.warning(f"Could not retrieve NVML/Driver version info: {e}")
            return True
        except pynvml.NVMLError as error:
            logger.error(f"Failed to initialize NVML: {error}", exc_info=True)
            _nvml_initialized = False
            return False


def shutdown_nvml():
    """Shuts down the NVML library (thread-safe)."""
    global _nvml_initialized, _gpu_handles
    with _nvml_lock:
        if not _nvml_initialized:
            logger.debug("NVML not initialized, skipping shutdown.")
            return
        try:
            pynvml.nvmlShutdown()
            _nvml_initialized = False
            _gpu_handles = {}  # Clear cached handles
            logger.info("NVML shut down successfully.")
        except pynvml.NVMLError as error:
            # Can sometimes raise an error if already shutdown elsewhere, log as warning
            logger.warning(
                f"NVML shutdown encountered an error (potentially already shut down): {error}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during NVML shutdown: {e}", exc_info=True)


def get_gpu_handle(device_index: int = 0) -> Optional[pynvml.c_nvmlDevice_t]:
    """
    Gets the NVML handle for a specific GPU device (thread-safe init).

    Args:
        device_index: The index of the GPU device.

    Returns:
        The NVML device handle, or None if NVML init fails or device not found.
    """
    # Ensure NVML is initialized before getting handle
    if not _nvml_initialized:
        if not init_nvml():
            logger.error("NVML not initialized. Cannot get GPU handle.")
            return None

    # No lock needed here assuming handle retrieval itself is safe after init
    if device_index in _gpu_handles:
        return _gpu_handles[device_index]

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        _gpu_handles[device_index] = handle
        logger.debug(f"Retrieved handle for GPU device {device_index}.")
        return handle
    except pynvml.NVMLError as error:
        logger.error(
            f"Failed to get handle for GPU device {device_index}: {error}",
            exc_info=True,
        )
        return None


def get_gpu_power_usage(handle: pynvml.c_nvmlDevice_t) -> Optional[float]:
    """Gets the current power usage of the GPU in Watts."""
    # (Implementation remains the same as before)
    if not handle:
        logger.error("Invalid GPU handle provided for power usage query.")
        return None
    try:
        power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_watts = power_milliwatts / 1000.0
        return power_watts
    except pynvml.NVMLError as error:
        if error.value == pynvml.NVMLError_NotSupported:
            logger.warning(f"Power usage reporting not supported for this GPU.")
        # NVMLError_GPUIsLost can happen sometimes
        elif error.value == pynvml.NVMLError_GPUIsLost:
            logger.error(
                f"GPU handle seems invalid (GPU lost?) during power query: {error}"
            )
        else:
            logger.error(f"Failed to get power usage: {error}", exc_info=True)
        return None


def get_gpu_memory_usage(
    handle: pynvml.c_nvmlDevice_t,
) -> Optional[Tuple[float, float, float]]:
    """Gets the memory usage of the GPU in MiB."""
    # (Implementation remains the same as before)
    if not handle:
        logger.error("Invalid GPU handle provided for memory usage query.")
        return None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        bytes_to_mib = 1 / (1024 * 1024)
        total_mib = mem_info.total * bytes_to_mib
        used_mib = mem_info.used * bytes_to_mib
        free_mib = mem_info.free * bytes_to_mib
        return used_mib, total_mib, free_mib
    except pynvml.NVMLError as error:
        logger.error(f"Failed to get memory usage: {error}", exc_info=True)
        return None


# --- NEW: Energy Monitoring Class ---
class GPUEnergyMonitor:
    """
    Monitors GPU energy consumption using background thread sampling.

    Usage:
        monitor = GPUEnergyMonitor(device_index=0, interval_sec=0.2)
        with monitor:
            # Run your GPU-intensive code here (e.g., training loop)
            time.sleep(5) # Simulate work
        total_energy = monitor.get_total_energy()
        if total_energy is not None:
            print(f"Total energy consumed: {total_energy:.2f} Joules")
    """

    def __init__(self, device_index: int = 0, interval_sec: float = 0.2):
        """
        Initializes the energy monitor.

        Args:
            device_index: The index of the GPU to monitor.
            interval_sec: Sampling interval in seconds.
        """
        self._device_index = device_index
        self._interval_sec = interval_sec
        self._handle: Optional[pynvml.c_nvmlDevice_t] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[Tuple[float, float]] = (
            []
        )  # List of (timestamp, power_watts)
        self._is_running = False
        self._total_energy_joules: Optional[float] = None

        # Ensure NVML is initialized and get handle
        if init_nvml():
            self._handle = get_gpu_handle(self._device_index)
            if self._handle is None:
                logger.error(
                    f"EnergyMonitor: Failed to get handle for GPU {self._device_index}. Monitoring disabled."
                )
        else:
            logger.error(
                "EnergyMonitor: NVML initialization failed. Monitoring disabled."
            )

    def _monitor_energy(self):
        """Target function for the background monitoring thread."""
        logger.debug("Energy monitoring thread started.")
        while not self._stop_event.is_set():
            start_sample_time = time.time()
            if self._handle:
                power_watts = get_gpu_power_usage(self._handle)
                current_time = time.time()  # Get time closer to power reading
                if power_watts is not None:
                    with (
                        _nvml_lock
                    ):  # Use lock if appending to shared list (though GIL might protect list append)
                        self._samples.append((current_time, power_watts))
                else:
                    # Optionally stop monitoring if power reading fails consistently
                    logger.warning("Failed to get power reading in monitoring thread.")

            # Adjust sleep time to maintain approximate interval
            elapsed = time.time() - start_sample_time
            sleep_time = max(0, self._interval_sec - elapsed)
            time.sleep(sleep_time)
        logger.debug("Energy monitoring thread stopped.")

    def _calculate_energy(self) -> Optional[float]:
        """Calculates total energy using the trapezoidal rule."""
        with _nvml_lock:  # Access samples safely
            if len(self._samples) < 2:
                logger.warning("Not enough samples collected to calculate energy.")
                return None

            total_energy = 0.0
            # Sort samples by timestamp just in case (though append should keep order)
            sorted_samples = sorted(self._samples, key=lambda x: x[0])

            for i in range(len(sorted_samples) - 1):
                t1, p1 = sorted_samples[i]
                t2, p2 = sorted_samples[i + 1]

                time_delta = t2 - t1
                if time_delta <= 0:  # Avoid division by zero or negative time
                    logger.warning(
                        f"Skipping energy calculation segment due to non-positive time delta: {time_delta}"
                    )
                    continue

                avg_power = (p1 + p2) / 2.0
                energy_segment = avg_power * time_delta  # Joules = Watts * seconds
                total_energy += energy_segment

            self._total_energy_joules = total_energy
            return total_energy

    def start(self):
        """Starts the energy monitoring thread."""
        if self._is_running:
            logger.warning("Energy monitor is already running.")
            return
        if not self._handle:
            logger.error("Cannot start energy monitor: No valid GPU handle.")
            return

        self._samples = []  # Clear previous samples
        self._stop_event.clear()
        self._total_energy_joules = None  # Reset calculated energy
        self._monitoring_thread = threading.Thread(
            target=self._monitor_energy, daemon=True
        )
        self._monitoring_thread.start()
        self._is_running = True
        logger.info(
            f"Started energy monitoring for GPU {self._device_index} (interval: {self._interval_sec}s)."
        )

    def stop(self) -> Optional[float]:
        """Stops the monitoring thread and calculates the total energy."""
        if not self._is_running:
            logger.warning("Energy monitor is not running.")
            return (
                self._total_energy_joules
            )  # Return previously calculated value if any

        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(
                timeout=self._interval_sec * 2 + 1
            )  # Wait for thread with timeout
            if self._monitoring_thread.is_alive():
                logger.warning("Energy monitoring thread did not stop gracefully.")
        self._is_running = False
        logger.info("Stopped energy monitoring.")

        # Calculate energy after stopping
        return self._calculate_energy()

    def get_total_energy(self) -> Optional[float]:
        """Returns the calculated total energy in Joules, or None if not calculated."""
        # This might return None if stop() wasn't called or calculation failed
        return self._total_energy_joules

    # --- Context Manager ---
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # Optional: Re-raise exception if one occurred within the 'with' block
        # return False # Returning False re-raises the exception


# Ensure NVML shutdown happens at program exit if not explicitly called
import atexit

atexit.register(shutdown_nvml)

if __name__ == "__main__":
    # Example Usage
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("Testing monitoring_utils including GPUEnergyMonitor...")

    if torch.cuda.is_available():
        try:
            print("\n--- Testing GPUEnergyMonitor ---")
            monitor = GPUEnergyMonitor(
                device_index=0, interval_sec=0.1
            )  # Sample every 100ms

            if monitor._handle:  # Check if monitor initialized correctly
                with monitor:
                    print("Monitoring started. Simulating work for 3 seconds...")
                    # Simulate GPU work (replace with actual GPU tasks)
                    a = torch.randn(10000, 10000).cuda()
                    for _ in range(5):
                        b = torch.matmul(a, a)
                        time.sleep(0.5)
                    del a, b
                    torch.cuda.empty_cache()
                    print("Simulated work finished.")

                total_j = monitor.get_total_energy()
                if total_j is not None:
                    print(f"Calculated Total Energy: {total_j:.2f} Joules")
                    print(f"Number of power samples collected: {len(monitor._samples)}")
                else:
                    print("Energy calculation failed (check logs/samples).")
            else:
                print(
                    "Skipping GPUEnergyMonitor test as GPU handle could not be obtained."
                )

        except Exception as e:
            print(f"Error during GPUEnergyMonitor test: {e}", exc_info=True)
        finally:
            # Explicit shutdown (though atexit should handle it too)
            shutdown_nvml()
    else:
        print("CUDA not available, skipping GPUEnergyMonitor test.")

    print("\nMonitoring utils testing finished.")
