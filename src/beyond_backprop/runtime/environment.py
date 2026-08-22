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
) -> RunMetadata:
    commit, dirty = _git_state()
    return RunMetadata(
        run_id=run_id,
        command_line=" ".join(shlex.quote(part) for part in sys.argv),
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        device=device,
        seed=seed,
        hostname=socket.gethostname(),
    )


def environment_dict(device: str = "cpu", seed: int | None = None) -> dict[str, Any]:
    metadata = collect_run_metadata("environment", device=device, seed=seed)
    return {
        "run_id": metadata.run_id,
        "timestamp_utc": metadata.timestamp_utc,
        "command_line": metadata.command_line,
        "git_commit": metadata.git_commit,
        "git_dirty": metadata.git_dirty,
        "python_version": metadata.python_version,
        "torch_version": metadata.torch_version,
        "device": metadata.device,
        "seed": metadata.seed,
        "hostname": metadata.hostname,
    }
