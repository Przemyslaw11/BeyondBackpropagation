"""Lazy construction of monitoring services from resolved configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config.models import ExperimentConfig
from ..runtime import resolve_device
from .carbon import CodeCarbonResourceMonitor
from .clock import WallClockResourceMonitor
from .composite import CompositeResourceMonitor
from .noop import NoOpResourceMonitor
from .nvml import NvmlResourceMonitor


def _mapping(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    return config.to_mapping() if isinstance(config, ExperimentConfig) else dict(config)


def build_resource_monitor(config: ExperimentConfig | Mapping[str, Any]):
    """Build a monitor without importing optional hardware packages eagerly."""

    mapping = _mapping(config)
    settings = mapping.get("monitoring", {})
    if not settings.get("enabled", False):
        return NoOpResourceMonitor()

    monitors: list[Any] = [WallClockResourceMonitor()]
    if settings.get("energy_enabled", False):
        device = resolve_device(mapping)
        if getattr(device, "type", str(device)) == "cuda":
            monitors.append(
                NvmlResourceMonitor(
                    device_index=int(settings.get("device_index", 0)),
                    interval_sec=float(settings.get("energy_interval_sec", 0.2)),
                )
            )
        else:
            monitors.append(
                NvmlResourceMonitor(
                    interval_sec=float(settings.get("energy_interval_sec", 0.2))
                )
            )

    carbon = mapping.get("carbon_tracker", {})
    if carbon.get("enabled", False):
        carbon_config = {
            key: value
            for key, value in carbon.items()
            if key
            in {
                "project_name",
                "output_dir",
                "country_iso_code",
                "log_level",
                "measure_power_secs",
            }
        }
        monitors.append(CodeCarbonResourceMonitor(**carbon_config))
    return CompositeResourceMonitor(monitors)


__all__ = ["build_resource_monitor"]
