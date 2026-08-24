"""Collection of reproducibility metadata without importing experiment code."""

from __future__ import annotations

import platform
import shlex
import socket
import subprocess
import sys
from typing import Any

import torch

from ..contracts import RunMetadata


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def collect_run_metadata(
    run_id: str,
    *,
    device: str = "cpu",
    seed: int | None = None,
    config_hash: str | None = None,
) -> RunMetadata:
    commit, dirty = _git_state()
    cuda_available = bool(torch.cuda.is_available())
    cuda_device: str | None = None
    cuda_driver_version: str | None = None
    if cuda_available:
        try:
            cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
        except (RuntimeError, AssertionError):
            cuda_device = None
        driver_version = getattr(torch.cuda, "driver_version", None)
        if driver_version is not None:
            cuda_driver_version = str(driver_version)
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(
        mps_backend is not None
        and callable(getattr(mps_backend, "is_available", None))
        and mps_backend.is_available()
    )
    torchvision_version: str | None
    try:
        from importlib.metadata import version

        torchvision_version = version("torchvision")
    except Exception:
        torchvision_version = None
    return RunMetadata(
        run_id=run_id,
        command_line=" ".join(shlex.quote(part) for part in sys.argv),
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torchvision_version=torchvision_version,
        cuda_available=cuda_available,
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        cuda_device=cuda_device,
        cuda_driver_version=cuda_driver_version,
        mps_available=mps_available,
        mps_device=platform.machine() if mps_available else None,
        device=device,
        seed=seed,
        config_hash=config_hash,
        hostname=socket.gethostname(),
    )


def environment_dict(
    device: str = "cpu", seed: int | None = None, config_hash: str | None = None
) -> dict[str, Any]:
    return collect_run_metadata(
        "environment", device=device, seed=seed, config_hash=config_hash
    ).to_dict()
