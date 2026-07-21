"""Backend policy helpers for local and Slurm-style execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover - allows import-time validation without torch.
    torch = None  # type: ignore[assignment]


def _mps_is_available() -> bool:
    """Return True when PyTorch can execute on Apple Silicon MPS."""
    if torch is None:
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    is_built = getattr(mps_backend, "is_built", None)
    return bool(
        callable(is_available)
        and callable(is_built)
        and is_available()
        and is_built()
    )


def _normalize_backend_name(name: str | None) -> str:
    backend_name = (name or "slurm").strip().lower()
    return backend_name if backend_name in {"local", "slurm"} else "slurm"


@dataclass(frozen=True)
class ExecutionBackend:
    """Base policy for runtime decisions that differ by execution environment."""

    name: str

    def resolve_device(self, device_preference: str = "auto") -> torch.device:
        """Resolve the PyTorch device for the backend."""
        if torch is None:
            raise RuntimeError("PyTorch is required to resolve execution devices.")
        preference = (device_preference or "auto").strip().lower()

        if preference == "cpu":
            return torch.device("cpu")
        if preference == "cuda":
            return (
                torch.device("cuda")
                if torch.cuda.is_available()
                else self._fallback_device()
            )
        if preference == "mps":
            return torch.device("mps") if _mps_is_available() else self._fallback_device()

        return self._fallback_device()

    def _fallback_device(self) -> torch.device:
        if torch is None:
            raise RuntimeError("PyTorch is required to resolve execution devices.")
        if self.name == "local":
            if _mps_is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def resolve_dataloader_defaults(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return backend-friendly DataLoader defaults."""
        if torch is None:
            return {"num_workers": 0, "pin_memory": False}

        defaults = (
            {"num_workers": 0, "pin_memory": False}
            if self.name == "local"
            else (
                {"num_workers": 4, "pin_memory": True}
                if torch.cuda.is_available()
                else {"num_workers": 0, "pin_memory": False}
            )
        )

        if not isinstance(config, dict):
            return defaults

        backend_config = config.get("backend", {}).get(self.name, {})
        loader_config = backend_config.get("data_loader", {})
        if loader_config:
            for key in ("num_workers", "pin_memory"):
                if key in loader_config and loader_config[key] is not None:
                    defaults[key] = loader_config[key]
            return defaults

        legacy_loader_config = config.get("data_loader", {})
        for key in ("num_workers", "pin_memory"):
            if key in legacy_loader_config and legacy_loader_config[key] is not None:
                defaults[key] = legacy_loader_config[key]
        return defaults

    def resolve_results_dir(self, config: Dict[str, Any]) -> str:
        """Resolve the default results directory for the backend."""
        backend_config = config.get("backend", {}).get(self.name, {})
        explicit_dir = backend_config.get("results_dir") or config.get("results", {}).get(
            "dir"
        )
        if explicit_dir:
            return explicit_dir
        return "results/local" if self.name == "local" else "results"

    def resolve_log_file(self, config: Dict[str, Any], experiment_name: str) -> str:
        """Resolve the default run log file path."""
        backend_config = config.get("backend", {}).get(self.name, {})
        log_config = config.get("logging", {})
        explicit_backend_log_file = backend_config.get("log_file")
        if explicit_backend_log_file:
            return explicit_backend_log_file
        explicit_log_file = log_config.get("log_file")
        if explicit_log_file:
            return explicit_log_file
        results_dir = self.resolve_results_dir(config)
        return os.path.join(results_dir, experiment_name, f"{experiment_name}_run.log")

    def prepare_environment(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Return environment variable overrides for the backend."""
        env_overrides: Dict[str, str] = {}
        if self.name == "local":
            env_overrides["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if not isinstance(config, dict):
            return env_overrides

        backend_config = config.get("backend", {}).get(self.name, {})
        env_overrides.update(backend_config.get("env", {}))
        return env_overrides


class SlurmBackend(ExecutionBackend):
    def __init__(self) -> None:
        super().__init__(name="slurm")


class LocalBackend(ExecutionBackend):
    def __init__(self) -> None:
        super().__init__(name="local")


def get_execution_backend(config: Dict[str, Any]) -> ExecutionBackend:
    """Construct the backend policy from the merged configuration."""
    general_config = config.get("general", {}) if isinstance(config, dict) else {}
    backend_name = _normalize_backend_name(general_config.get("backend"))
    if backend_name == "local":
        return LocalBackend()
    return SlurmBackend()