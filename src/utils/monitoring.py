# File: src/utils/monitoring.py
import torch
import pynvml
import time
import logging
import threading
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger(__name__)

# --- NVML Initialization & Management ---
_nvml_initialized = False
_nvml_lock = threading.Lock()  # Lock for thread-safe init/shutdown
_gpu_handles: Dict[int, pynvml.c_nvmlDevice_t] = {}


def init_nvml() -> bool:  # Added return type hint
    """Initializes the NVML library if not already done (thread-safe). Returns True if successful or already initialized."""
    global _nvml_initialized
    # Quick check without lock first for performance
    if _nvml_initialized:
        return True
    with _nvml_lock:
        # Double check after acquiring lock
        if _nvml_initialized:
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
        except pynvml.NVMLError_LibraryNotFound:
            logger.error(
                "NVML library not found. NVIDIA driver may not be installed or pynvml installation issue."
            )
            _nvml_initialized = False
            return False
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
            # NVMLError_Uninitialized can happen if already shut down
            if error.value == pynvml.NVMLError_Uninitialized:
                logger.debug("NVML already shut down.")
                _nvml_initialized = False  # Ensure state is correct
            else:
                logger.warning(f"NVML shutdown encountered an error: {error}")
        except Exception as e:
            logger.error(f"Unexpected error during NVML shutdown: {e}", exc_info=True)


def get_gpu_handle(device_index: int = 0) -> Optional[pynvml.c_nvmlDevice_t]:
    """
    Gets the NVML handle for a specific GPU device (initializes NVML if needed).

    Args:
        device_index: The index of the GPU device.

    Returns:
        The NVML device handle, or None if NVML init fails or device not found.
    """
    # Ensure NVML is initialized before getting handle
    if not _nvml_initialized:
        if not init_nvml():
            logger.error("NVML initialization failed. Cannot get GPU handle.")
            return None

    # Check cache first (no lock needed for read assuming dict access is atomic)
    if device_index in _gpu_handles:
        return _gpu_handles[device_index]

    # Get handle and cache it (potential race condition if multiple threads request same new handle, but unlikely to be harmful)
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        # Use lock for writing to shared cache
        with _nvml_lock:
            _gpu_handles[device_index] = handle
        logger.debug(f"Retrieved and cached handle for GPU device {device_index}.")
        return handle
    except pynvml.NVMLError_InvalidArg:
        logger.error(f"GPU device index {device_index} is invalid or out of range.")
        return None
    except pynvml.NVMLError as error:
        logger.error(
            f"Failed to get handle for GPU device {device_index}: {error}",
            exc_info=True,
        )
        return None


def get_gpu_power_usage(handle: pynvml.c_nvmlDevice_t) -> Optional[float]:
    """Gets the current power usage of the GPU in Watts."""
    if not handle:
        logger.debug("Invalid GPU handle provided for power usage query.")
        return None
    if not _nvml_initialized:
        logger.warning("NVML not initialized, cannot get power usage.")
        return None
    try:
        power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_watts = power_milliwatts / 1000.0
        return power_watts
    except pynvml.NVMLError as error:
        if error.value == pynvml.NVMLError_NotSupported:
            logger.warning(f"Power usage reporting not supported for this GPU handle.")
        elif error.value == pynvml.NVMLError_GPUIsLost:
            logger.error(f"GPU lost during power query.")
        elif error.value == pynvml.NVMLError_Uninitialized:
            logger.error("NVML uninitialized during power query.")
        else:
            logger.error(f"Failed to get power usage: {error}", exc_info=True)
        return None


def get_gpu_memory_usage(
    handle: pynvml.c_nvmlDevice_t,
) -> Optional[Tuple[float, float, float]]:
    """
    Gets the memory usage of the GPU in MiB (Used, Total, Free).

    Returns:
        Tuple[Used MiB, Total MiB, Free MiB] or None if failed.
    """
    if not handle:
        logger.debug("Invalid GPU handle provided for memory usage query.")
        return None
    if not _nvml_initialized:
        logger.warning("NVML not initialized, cannot get memory usage.")
        return None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Use 1024*1024 for MiB (Mebibytes)
        bytes_to_mib = 1 / (1024**2)
        total_mib = mem_info.total * bytes_to_mib
        used_mib = mem_info.used * bytes_to_mib
        free_mib = mem_info.free * bytes_to_mib
        return used_mib, total_mib, free_mib
    except pynvml.NVMLError as error:
        logger.error(f"Failed to get memory usage: {error}", exc_info=True)
        return None


# --- Energy Monitoring Class ---
class GPUEnergyMonitor:
    """
    Monitors GPU energy consumption using background thread sampling.

    Usage:
        monitor = GPUEnergyMonitor(device_index=0, interval_sec=0.2)
        with monitor:
            # Run your GPU-intensive code here (e.g., training loop)
            time.sleep(5) # Simulate work
        total_energy = monitor.get_total_energy_joules() # Use specific getter
        if total_energy is not None:
            print(f"Total energy consumed: {total_energy:.2f} Joules")
    """

    def __init__(self, device_index: int = 0, interval_sec: float = 0.2):
        """
        Initializes the energy monitor.

        Args:
            device_index: The index of the GPU to monitor.
            interval_sec: Sampling interval in seconds. Must be positive.
        """
        if interval_sec <= 0:
            raise ValueError("Sampling interval must be positive.")

        self._device_index = device_index
        self._interval_sec = interval_sec
        self._handle: Optional[pynvml.c_nvmlDevice_t] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[Tuple[float, float]] = (
            []
        )  # List of (timestamp, power_watts)
        self._samples_lock = threading.Lock()  # Lock for accessing samples list
        self._is_running = False
        self._total_energy_joules: Optional[float] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

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
        last_sample_time = time.time()  # Track time for interval accuracy

        while not self._stop_event.is_set():
            current_time = time.time()

            # --- Sample Power ---
            power_watts = None
            if self._handle:
                power_watts = get_gpu_power_usage(self._handle)

            # --- Store Sample (even if None, to mark time) ---
            if power_watts is not None:
                with self._samples_lock:
                    self._samples.append((current_time, power_watts))
            else:
                # Store timestamp with None power if reading fails
                with self._samples_lock:
                    self._samples.append((current_time, None))
                if self._handle:  # Only log warning if handle was valid
                    logger.warning("Failed to get power reading in monitoring thread.")

            # --- Sleep Logic ---
            # Calculate time elapsed since last intended sample time
            elapsed_since_last_intended = current_time - last_sample_time
            # Calculate desired next sample time
            next_intended_sample_time = last_sample_time + self._interval_sec
            # Calculate sleep duration needed to reach next intended time
            sleep_duration = next_intended_sample_time - time.time()

            if sleep_duration > 0:
                time.sleep(sleep_duration)
                last_sample_time = (
                    next_intended_sample_time  # Update base for next interval
                )
            else:
                # Sampling took longer than interval, don't sleep, update last_sample_time
                logger.debug(
                    f"Energy sampling took longer than interval: {elapsed_since_last_intended:.4f}s"
                )
                last_sample_time = current_time

        logger.debug("Energy monitoring thread stopped.")

    def _calculate_energy(self) -> Optional[float]:
        """Calculates total energy using the trapezoidal rule, handling None values."""
        with self._samples_lock:  # Access samples safely
            valid_samples = [(t, p) for t, p in self._samples if p is not None]

            if len(valid_samples) < 2:
                logger.warning(
                    "Not enough valid power samples collected to calculate energy."
                )
                return None

            total_energy = 0.0
            # Ensure samples are sorted by time (should be, but safety check)
            sorted_samples = sorted(valid_samples, key=lambda x: x[0])

            for i in range(len(sorted_samples) - 1):
                t1, p1 = sorted_samples[i]
                t2, p2 = sorted_samples[i + 1]

                time_delta = t2 - t1
                if time_delta <= 0:
                    logger.warning(
                        f"Skipping energy segment due to non-positive time delta: {time_delta}"
                    )
                    continue

                # Simple trapezoidal rule: avg power * time duration
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

        with self._samples_lock:
            self._samples = []  # Clear previous samples
        self._stop_event.clear()
        self._total_energy_joules = None  # Reset calculated energy
        self._start_time = time.time()  # Record start time
        self._end_time = None

        self._monitoring_thread = threading.Thread(
            target=self._monitor_energy,
            daemon=True,  # Daemon thread exits if main program exits
        )
        self._monitoring_thread.start()
        self._is_running = True
        logger.info(
            f"Started energy monitoring for GPU {self._device_index} (interval: {self._interval_sec}s)."
        )

    def stop(self) -> Optional[float]:
        """Stops the monitoring thread and calculates the total energy. Returns energy in Joules."""
        if not self._is_running:
            logger.info("Energy monitor is not running.")
            return (
                self._total_energy_joules
            )  # Return previously calculated value if any

        self._stop_event.set()
        self._end_time = time.time()  # Record end time
        if self._monitoring_thread:
            # Wait briefly for the thread to finish processing last sample and exit
            self._monitoring_thread.join(timeout=self._interval_sec * 2 + 1)
            if self._monitoring_thread.is_alive():
                logger.warning("Energy monitoring thread did not stop gracefully.")
        self._is_running = False
        logger.info("Stopped energy monitoring.")

        # Calculate energy after stopping
        return self._calculate_energy()

    def get_total_energy_joules(self) -> Optional[float]:
        """Returns the calculated total energy in Joules, or None if not calculated."""
        # Ensures calculation happens if monitor stopped but energy not yet calculated
        if (
            self._total_energy_joules is None
            and not self._is_running
            and self._start_time is not None
        ):
            self._calculate_energy()
        return self._total_energy_joules

    def get_monitoring_duration(self) -> Optional[float]:
        """Returns the duration the monitor was active in seconds."""
        if self._start_time is None:
            return None
        end_time = self._end_time if self._end_time is not None else time.time()
        return end_time - self._start_time

    # --- Context Manager ---
    def __enter__(self):
        """Starts monitoring when entering the 'with' block."""
        self.start()
        return self  # Return the monitor instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops monitoring when exiting the 'with' block."""
        self.stop()
        # Optional: Re-raise exception if one occurred within the 'with' block
        # return False # Returning False (or None implicitly) re-raises the exception
        # return True # Suppress any exception that occurred within the block


# Ensure NVML shutdown happens at program exit
import atexit

atexit.register(shutdown_nvml)
