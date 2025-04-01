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


def init_nvml() -> bool:
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
            # Attempt to initialize NVML
            pynvml.nvmlInit()
            _nvml_initialized = True
            logger.info("NVML initialized successfully.")

            # Log driver and NVML versions only once (more robustly)
            try:
                # Attempt to get driver version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    # Ensure driver_version is decoded if it's bytes (common in older pynvml)
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')
                    logger.info(f"NVIDIA Driver Version: {driver_version}")
                except AttributeError:
                    logger.warning("pynvml.nvmlSystemGetDriverVersion() not found in this pynvml version.")
                except pynvml.NVMLError as e:
                    logger.warning(f"Could not retrieve Driver version info (NVML Error): {e}")

                # Attempt to get NVML version
                try:
                    nvml_version = pynvml.nvmlSystemGetNvmlVersion()
                    # Ensure nvml_version is decoded if it's bytes
                    if isinstance(nvml_version, bytes):
                        nvml_version = nvml_version.decode('utf-8')
                    logger.info(f"NVML Library Version: {nvml_version}")
                except AttributeError:
                    logger.warning("pynvml.nvmlSystemGetNvmlVersion() not found in this pynvml version.")
                except pynvml.NVMLError as e:
                    logger.warning(f"Could not retrieve NVML version info (NVML Error): {e}")

            except Exception as e: # Catch any other unexpected error during version logging
                logger.warning(f"An unexpected error occurred while retrieving NVML/Driver version info: {e}")

            # Return True because nvmlInit() succeeded, even if version logging failed
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
        except Exception as e: # Catch potential unexpected errors during init itself
             logger.error(f"An unexpected error occurred during NVML initialization: {e}", exc_info=True)
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
            # Don't log excessive errors if power reading fails often
            # Consider logging less frequently or only once
            pass  # logger.error(f"Failed to get power usage: {error}")
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
    """

    def __init__(self, device_index: int = 0, interval_sec: float = 0.2):
        if interval_sec <= 0:
            raise ValueError("Sampling interval must be positive.")

        self._device_index = device_index
        self._interval_sec = interval_sec
        self._handle: Optional[pynvml.c_nvmlDevice_t] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[Tuple[float, Optional[float]]] = (
            []
        )  # Store None if reading fails
        self._samples_lock = threading.Lock()
        self._is_running = False
        self._total_energy_joules: Optional[float] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._power_error_logged = False  # Flag to log power reading error only once

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
        logger.debug(f"Energy monitoring thread started for GPU {self._device_index}.")
        last_sample_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()
            power_watts = None
            if self._handle:
                power_watts = get_gpu_power_usage(self._handle)
                if power_watts is None and not self._power_error_logged:
                    logger.warning(
                        f"EnergyMonitor: Failed to get power reading for GPU {self._device_index}. Will store None."
                    )
                    self._power_error_logged = True  # Log only once per start

            # Store timestamp and power (even if None)
            with self._samples_lock:
                self._samples.append((current_time, power_watts))

            # Sleep accurately
            elapsed = time.time() - last_sample_time
            sleep_duration = self._interval_sec - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            # Update last_sample_time based on when it *should* have woken up
            last_sample_time += self._interval_sec

        logger.debug(f"Energy monitoring thread stopped for GPU {self._device_index}.")

    def _calculate_energy(self) -> Optional[float]:
        """Calculates total energy using the trapezoidal rule, handling None values."""
        with self._samples_lock:
            if len(self._samples) < 2:
                logger.warning(
                    f"EnergyMonitor GPU {self._device_index}: Not enough samples ({len(self._samples)}) collected to calculate energy."
                )
                return None

            total_energy = 0.0
            # Ensure samples are sorted (should be by design)
            # sorted_samples = sorted(self._samples, key=lambda x: x[0]) # Usually not needed

            for i in range(len(self._samples) - 1):
                t1, p1 = self._samples[i]
                t2, p2 = self._samples[i + 1]

                time_delta = t2 - t1
                if time_delta <= 0:
                    # This might happen with timer precision issues, skip segment
                    continue

                # Handle None power values: assume power is constant from the last valid reading
                # or average of neighbors if both are valid. Use 0 if no valid reading available.
                p1_valid = p1 if p1 is not None else (p2 if p2 is not None else 0.0)
                p2_valid = p2 if p2 is not None else (p1 if p1 is not None else 0.0)

                if p1 is None and p2 is None:
                    avg_power = 0.0  # Or potentially use a running average? Simple approach for now.
                    if (
                        not self._power_error_logged
                    ):  # Log warning if power reading failed for segment
                        logger.warning(
                            f"EnergyMonitor GPU {self._device_index}: Missing power data for time segment [{t1:.2f}, {t2:.2f}]. Assuming 0W."
                        )
                        self._power_error_logged = True  # Log only once
                else:
                    avg_power = (p1_valid + p2_valid) / 2.0

                energy_segment = avg_power * time_delta
                total_energy += energy_segment

            if total_energy == 0 and self._power_error_logged:
                logger.warning(
                    f"EnergyMonitor GPU {self._device_index}: Calculated total energy is 0 Joules, likely due to persistent power reading failures."
                )

            self._total_energy_joules = total_energy
            return total_energy

    def start(self):
        """Starts the energy monitoring thread."""
        if self._is_running:
            logger.warning(
                f"Energy monitor for GPU {self._device_index} is already running."
            )
            return
        if not self._handle:
            logger.error(
                f"Cannot start energy monitor for GPU {self._device_index}: No valid GPU handle."
            )
            return

        with self._samples_lock:
            self._samples = []
        self._stop_event.clear()
        self._total_energy_joules = None
        self._start_time = time.time()
        self._end_time = None
        self._power_error_logged = False  # Reset error log flag

        self._monitoring_thread = threading.Thread(
            target=self._monitor_energy,
            daemon=True,
        )
        self._monitoring_thread.start()
        self._is_running = True
        logger.info(
            f"Started energy monitoring for GPU {self._device_index} (interval: {self._interval_sec}s)."
        )

    def stop(self) -> Optional[float]:
        """Stops the monitoring thread and calculates the total energy. Returns energy in Joules."""
        if not self._is_running:
            logger.info(f"Energy monitor for GPU {self._device_index} is not running.")
            return self._total_energy_joules

        self._stop_event.set()
        self._end_time = time.time()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=self._interval_sec * 2 + 1)
            if self._monitoring_thread.is_alive():
                logger.warning(
                    f"Energy monitoring thread for GPU {self._device_index} did not stop gracefully."
                )
        self._is_running = False
        logger.info(f"Stopped energy monitoring for GPU {self._device_index}.")

        return self._calculate_energy()

    def get_total_energy_joules(self) -> Optional[float]:
        """Returns the calculated total energy in Joules, or None if not calculated."""
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Ensure NVML shutdown happens at program exit
import atexit

atexit.register(shutdown_nvml)
