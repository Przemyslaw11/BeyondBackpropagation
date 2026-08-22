from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _run(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_legacy_single_wrapper_translates_validation_and_missing_config() -> None:
    result = _run("python3", "scripts/run_experiment.py", "--config", "missing.yaml")
    assert result.returncode == 2
    assert "Configuration error" in result.stderr


def test_local_array_wrapper_rejects_empty_directory(tmp_path) -> None:
    result = _run(
        "python3",
        "scripts/run_local_array.py",
        "--config-dir",
        str(tmp_path),
        "--dry-run",
    )
    assert result.returncode == 2


def test_slurm_wrappers_have_no_personal_account_defaults_and_validate_indices() -> (
    None
):
    array_script = ROOT / "scripts/slurm_scripts/run_array.slurm"
    text = array_script.read_text()
    assert "plgoncotherapy" not in text
    assert "plgrid-gpu-a100" not in text
    result = _run(
        "bash",
        str(array_script),
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "999", "SRUN_BIN": "missing-srun"},
    )
    assert result.returncode == 2
    assert "invalid SLURM_ARRAY_TASK_ID" in result.stderr


def test_slurm_single_propagates_srun_failure(tmp_path) -> None:
    fake_srun = tmp_path / "srun"
    fake_srun.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake_srun.chmod(0o755)
    result = _run(
        "bash",
        "scripts/slurm_scripts/run_single_experiment.slurm",
        "configs/mf/mnist_mlp_2x1000.yaml",
        env={**os.environ, "SRUN_BIN": str(fake_srun)},
    )
    assert result.returncode == 7
