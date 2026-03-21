"""GPU monitoring utilities using the pynvml library."""

import atexit
import csv
import logging
import os
import threading
import time
from types import TracebackType
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import psutil
import pynvml

logger = logging.getLogger(__name__)

_nvml_initialized = False
_nvml_lock = threading.Lock()
_gpu_handles: Dict[int, pynvml.c_nvmlDevice_t] = {}


class MonitorSample(NamedTuple):
    """One instant of the hardware time series written to the per-run CSV."""

    timestamp_sec: float
    power_watts: Optional[float]
    gpu_util_percent: Optional[int]
    mem_util_percent: Optional[int]
    gpu_mem_used_mib: Optional[float]
    gpu_temp_celsius: Optional[int]
    sm_clock_mhz: Optional[int]
    compute_processes: Optional[int]
    process_rss_mib: Optional[float]


def _log_nvml_version_info() -> None:
    """Logs the NVIDIA driver and NVML library versions if available."""
    try:
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")
        logger.info(f"NVIDIA Driver Version: {driver_version}")
    except AttributeError:
        logger.warning("pynvml.nvmlSystemGetDriverVersion() attr not found.")
    except pynvml.NVMLError as e:
        logger.warning(f"Could not get Driver version (NVML Error): {e}")

    try:
        nvml_version = pynvml.nvmlSystemGetNvmlVersion()
        if isinstance(nvml_version, bytes):
            nvml_version = nvml_version.decode("utf-8")
        logger.info(f"NVML Library Version: {nvml_version}")
    except AttributeError:
        logger.info("NVML version query is unavailable in this pynvml library.")
    except pynvml.NVMLError as e:
        logger.warning(f"Could not get NVML version (NVML Error): {e}")


def init_nvml() -> bool:
    """Initializes NVML library (thread-safe). Returns True if successful."""
    global _nvml_initialized
    if _nvml_initialized:
        return True
    with _nvml_lock:
        if _nvml_initialized:
            return True
        try:
            pynvml.nvmlInit()
            _nvml_initialized = True
            logger.info("NVML initialized successfully.")
            _log_nvml_version_info()
            return True
        except pynvml.NVMLError_LibraryNotFound:
            logger.error("NVML lib not found. NVIDIA driver may not be installed.")
            _nvml_initialized = False
            return False
        except pynvml.NVMLError as error:
            logger.error(f"Failed to initialize NVML: {error}", exc_info=True)
            _nvml_initialized = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error during NVML init: {e}", exc_info=True)
            _nvml_initialized = False
            return False


def shutdown_nvml() -> None:
    """Shuts down the NVML library (thread-safe)."""
    global _nvml_initialized, _gpu_handles
    with _nvml_lock:
        if not _nvml_initialized:
            logger.debug("NVML not initialized, skipping shutdown.")
            return
        try:
            pynvml.nvmlShutdown()
            _nvml_initialized = False
            _gpu_handles = {}
            logger.info("NVML shut down successfully.")
        except pynvml.NVMLError as error:
            if error.value == pynvml.NVMLError_Uninitialized:
                logger.debug("NVML already shut down.")
                _nvml_initialized = False
            else:
                logger.warning(f"NVML shutdown encountered an error: {error}")
        except Exception as e:
            logger.error(f"Unexpected error during NVML shutdown: {e}", exc_info=True)


def get_gpu_handle(device_index: int = 0) -> Optional[pynvml.c_nvmlDevice_t]:
    """Gets the NVML handle for a specific GPU device (initializes NVML if needed).

    Args:
        device_index: The index of the GPU device.

    Returns:
        The NVML device handle, or None if NVML init fails or device not found.
    """
    if not _nvml_initialized and not init_nvml():
        logger.error("NVML initialization failed. Cannot get GPU handle.")
        return None

    if device_index in _gpu_handles:
        return _gpu_handles[device_index]

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
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
            logger.warning("Power usage reporting not supported for this GPU handle.")
        elif error.value == pynvml.NVMLError_GPUIsLost:
            logger.error("GPU lost during power query.")
        elif error.value == pynvml.NVMLError_Uninitialized:
            logger.error("NVML uninitialized during power query.")
        else:
            pass
        return None


def get_gpu_memory_usage(
    handle: pynvml.c_nvmlDevice_t,
) -> Optional[Tuple[float, float, float]]:
    """Gets the memory usage of the GPU in MiB (Used, Total, Free).

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
        bytes_to_mib = 1 / (1024**2)
        total_mib = mem_info.total * bytes_to_mib
        used_mib = mem_info.used * bytes_to_mib
        free_mib = mem_info.free * bytes_to_mib
        return used_mib, total_mib, free_mib
    except pynvml.NVMLError as error:
        logger.error(f"Failed to get memory usage: {error}", exc_info=True)
        return None


def get_gpu_utilization(
    handle: pynvml.c_nvmlDevice_t,
) -> Optional[Tuple[int, int]]:
    """Gets (SM utilisation, memory-bandwidth utilisation) as percentages."""
    if not handle or not _nvml_initialized:
        return None
    try:
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return int(rates.gpu), int(rates.memory)
    except pynvml.NVMLError:
        return None


def get_gpu_temperature(handle: pynvml.c_nvmlDevice_t) -> Optional[int]:
    """Gets the GPU core temperature in degrees Celsius."""
    if not handle or not _nvml_initialized:
        return None
    try:
        return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except pynvml.NVMLError:
        return None


def get_gpu_sm_clock_mhz(handle: pynvml.c_nvmlDevice_t) -> Optional[int]:
    """Gets the current SM clock frequency in MHz."""
    if not handle or not _nvml_initialized:
        return None
    try:
        return int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    except pynvml.NVMLError:
        return None


def get_gpu_compute_process_count(handle: pynvml.c_nvmlDevice_t) -> Optional[int]:
    """Counts compute contexts on the GPU, exposing contention from other jobs."""
    if not handle or not _nvml_initialized:
        return None
    try:
        return len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
    except pynvml.NVMLError:
        return None


class GPUEnergyMonitor:
    """Monitors GPU energy consumption using background thread sampling.

    MODIFIED: Added check for non-positive time delta during energy calculation.
    """

    def __init__(
        self,
        device_index: int = 0,
        interval_sec: float = 0.2,
        csv_path: Optional[str] = None,
    ) -> None:
        """Initializes the GPUEnergyMonitor.

        Args:
            device_index: The index of the GPU to monitor.
            interval_sec: The sampling interval in seconds.
            csv_path: Where to write the per-run time series, or None to skip it.
        """
        if interval_sec <= 0:
            raise ValueError("Sampling interval must be positive.")

        self._device_index = device_index
        self._interval_sec = interval_sec
        self._csv_path = csv_path
        self._process = psutil.Process(os.getpid())
        self._handle: Optional[pynvml.c_nvmlDevice_t] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[MonitorSample] = []
        self._samples_lock = threading.Lock()
        self._is_running = False
        self._total_energy_joules: Optional[float] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._power_error_logged = False
        self._time_delta_error_logged = False

        if init_nvml():
            self._handle = get_gpu_handle(self._device_index)
            if self._handle is None:
                logger.error(
                    "EnergyMonitor for GPU %s disabled: could not get handle.",
                    self._device_index,
                )
        else:
            logger.error("EnergyMonitor disabled: NVML initialization failed.")

    def _take_sample(self, timestamp: float) -> MonitorSample:
        """Reads every instrumented counter once, tolerating unsupported queries."""
        power_watts = None
        utilization: Optional[Tuple[int, int]] = None
        memory: Optional[Tuple[float, float, float]] = None
        temperature = None
        sm_clock = None
        compute_processes = None

        if self._handle:
            power_watts = get_gpu_power_usage(self._handle)
            if power_watts is None and not self._power_error_logged:
                logger.warning(
                    "EnergyMonitor GPU %s: could not get power reading.",
                    self._device_index,
                )
                self._power_error_logged = True
            utilization = get_gpu_utilization(self._handle)
            memory = get_gpu_memory_usage(self._handle)
            temperature = get_gpu_temperature(self._handle)
            sm_clock = get_gpu_sm_clock_mhz(self._handle)
            compute_processes = get_gpu_compute_process_count(self._handle)

        try:
            rss_mib = self._process.memory_info().rss / (1024**2)
        except psutil.Error:
            rss_mib = None

        return MonitorSample(
            timestamp_sec=timestamp,
            power_watts=power_watts,
            gpu_util_percent=utilization[0] if utilization else None,
            mem_util_percent=utilization[1] if utilization else None,
            gpu_mem_used_mib=memory[0] if memory else None,
            gpu_temp_celsius=temperature,
            sm_clock_mhz=sm_clock,
            compute_processes=compute_processes,
            process_rss_mib=rss_mib,
        )

    def _monitor_energy(self) -> None:
        logger.debug(f"Energy monitoring thread started for GPU {self._device_index}.")
        last_sample_time = time.monotonic()

        while not self._stop_event.is_set():
            sample = self._take_sample(time.monotonic())

            with self._samples_lock:
                self._samples.append(sample)

            next_sample_time = last_sample_time + self._interval_sec
            sleep_duration = next_sample_time - time.monotonic()

            if sleep_duration > 0:
                self._stop_event.wait(timeout=sleep_duration)
                if self._stop_event.is_set():
                    break
            last_sample_time = next_sample_time

        logger.debug(f"Energy monitoring thread stopped for GPU {self._device_index}.")

    def _write_csv(self) -> None:
        """Writes the sampled time series so per-run traces survive the process."""
        if not self._csv_path:
            return
        with self._samples_lock:
            samples = list(self._samples)
        if not samples:
            logger.warning("No monitoring samples collected; skipping CSV write.")
            return

        origin = samples[0].timestamp_sec
        directory = os.path.dirname(self._csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(MonitorSample._fields)
                for sample in samples:
                    row = sample._replace(
                        timestamp_sec=sample.timestamp_sec - origin
                    )
                    writer.writerow(row)
            logger.info(
                f"Wrote {len(samples)} monitoring samples to {self._csv_path}."
            )
        except OSError as error:
            logger.error(f"Failed to write monitoring CSV: {error}")

    def get_peak(self, field: str) -> Optional[float]:
        """Returns the maximum recorded value of a MonitorSample field."""
        with self._samples_lock:
            values = [
                getattr(sample, field)
                for sample in self._samples
                if getattr(sample, field) is not None
            ]
        return max(values) if values else None

    def _calculate_energy(self) -> Optional[float]:
        """Calculates total energy using the trapezoidal rule, handling None values."""
        with self._samples_lock:
            if len(self._samples) < 2:
                logger.warning(
                    f"EnergyMonitor GPU {self._device_index}: Not enough samples "
                    f"({len(self._samples)}) to calculate energy."
                )
                return None

            total_energy = 0.0
            for i in range(len(self._samples) - 1):
                t1, p1 = self._samples[i].timestamp_sec, self._samples[i].power_watts
                t2, p2 = (
                    self._samples[i + 1].timestamp_sec,
                    self._samples[i + 1].power_watts,
                )

                time_delta = t2 - t1
                if time_delta <= 0:
                    if not self._time_delta_error_logged:
                        log_msg = (
                            f"EnergyMonitor GPU {self._device_index}: Non-positive "
                            f"time delta ({time_delta:.4f}s). Skipping."
                        )
                        logger.warning(log_msg)
                        self._time_delta_error_logged = True
                    continue
                p1_valid = p1 if p1 is not None else (p2 if p2 is not None else 0.0)
                p2_valid = p2 if p2 is not None else (p1 if p1 is not None else 0.0)

                if p1 is None and p2 is None:
                    avg_power = 0.0
                    if not self._power_error_logged:
                        log_msg = (
                            f"EnergyMonitor GPU {self._device_index}: "
                            f"Missing power data for segment [{t1:.2f}, {t2:.2f}]."
                        )
                        logger.warning(log_msg)
                        self._power_error_logged = True
                else:
                    avg_power = (p1_valid + p2_valid) / 2.0

                energy_segment = avg_power * time_delta
                total_energy += energy_segment

            if total_energy == 0 and self._power_error_logged:
                logger.warning(
                    "EnergyMonitor GPU %s: Total energy is 0 Joules, "
                    "likely due to persistent power reading failures.",
                    self._device_index,
                )

            self._total_energy_joules = total_energy
            return total_energy

    def start(self) -> None:
        """Starts the energy monitoring thread."""
        if self._is_running:
            logger.warning(
                f"Energy monitor for GPU {self._device_index} is already running."
            )
            return
        if not self._handle:
            logger.error(
                f"Cannot start monitor for GPU {self._device_index}: no valid handle."
            )
            return

        with self._samples_lock:
            self._samples = []
        self._stop_event.clear()
        self._total_energy_joules = None
        self._start_time = time.monotonic()
        self._end_time = None
        self._power_error_logged = False
        self._time_delta_error_logged = False

        self._monitoring_thread = threading.Thread(
            target=self._monitor_energy,
            daemon=True,
        )
        self._monitoring_thread.start()
        self._is_running = True
        logger.info(
            f"Started energy monitoring for GPU {self._device_index} "
            f"(interval: {self._interval_sec}s)."
        )

    def stop(self) -> Optional[float]:
        """Stops monitoring thread, calculates total energy. Returns Joules."""
        if not self._is_running:
            logger.info(f"Energy monitor for GPU {self._device_index} is not running.")
            return self._total_energy_joules

        self._stop_event.set()
        self._end_time = time.monotonic()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=self._interval_sec * 2 + 1)
            if self._monitoring_thread.is_alive():
                logger.warning(
                    "Energy monitor for GPU %s did not exit gracefully.",
                    self._device_index,
                )
        self._is_running = False
        logger.info(f"Stopped energy monitoring for GPU {self._device_index}.")

        self._write_csv()
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
        end_time = self._end_time if self._end_time is not None else time.monotonic()
        return end_time - self._start_time

    def __enter__(self) -> "GPUEnergyMonitor":
        """Starts the monitor when entering a context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stops the monitor when exiting a context manager."""
        self.stop()


atexit.register(shutdown_nvml)
