from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


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


def _write_stub_python(tmp_path: Path, manifest_lines: list[str] | None) -> Path:
    """Create a PYTHON_BIN stub that logs invocations and optionally fakes
    manifest generation by writing ``manifest_lines`` to the requested target."""
    stub = tmp_path / "stub_python.sh"
    lines = "\n".join(manifest_lines) + "\n" if manifest_lines is not None else ""
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_LOG"\n'
        'if [[ "$1" == "-" ]]; then\n'
        f'    printf "{lines}" > "$3"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_slurm_wrappers_have_no_personal_account_defaults_and_validate_indices(
    tmp_path,
) -> None:
    array_script = ROOT / "scripts/slurm_scripts/run_array.slurm"
    text = array_script.read_text()
    assert "plgoncotherapy" not in text
    assert "plgrid-gpu-a100" not in text
    manifest = tmp_path / "provided-manifest.txt"
    manifest.write_text("does-not-matter-a.yaml\ndoes-not-matter-b.yaml\n")
    result = _run(
        "bash",
        str(array_script),
        env={
            **os.environ,
            "CONFIG_MANIFEST": str(manifest),
            "SLURM_ARRAY_TASK_ID": "999",
            "SRUN_BIN": "missing-srun",
        },
    )
    assert result.returncode == 2
    assert "invalid SLURM_ARRAY_TASK_ID" in result.stderr


def test_slurm_array_consumes_explicit_manifest_without_regeneration(tmp_path) -> None:
    config = ROOT / "configs/mf/mnist_mlp_2x1000.yaml"
    manifest = tmp_path / "provided-manifest.txt"
    manifest.write_text(f"{config}\n")
    stub = _write_stub_python(tmp_path, manifest_lines=None)

    result = _run(
        "bash",
        "scripts/slurm_scripts/run_array.slurm",
        env={
            **os.environ,
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "CONFIG_MANIFEST": str(manifest),
            "SLURM_ARRAY_TASK_ID": "1",
            "PYTHON_BIN": str(stub),
            "SRUN_BIN": "missing-srun",
            "STUB_LOG": str(tmp_path / "stub_log.txt"),
        },
    )

    assert result.returncode == 0
    log = (tmp_path / "stub_log.txt").read_text()
    assert log.count("\n") == 1, f"expected exactly one invocation, got: {log}"
    assert f"--config {config}" in log
    assert manifest.read_text() == f"{config}\n"


def test_slurm_array_generates_job_scoped_manifest_atomically(tmp_path) -> None:
    first = ROOT / "configs/mf/mnist_mlp_2x1000.yaml"
    second = ROOT / "configs/ff/mnist_mlp_4x2000.yaml"
    stub = _write_stub_python(tmp_path, manifest_lines=[str(first), str(second)])

    result = _run(
        "bash",
        "scripts/slurm_scripts/run_array.slurm",
        env={
            **os.environ,
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "SLURM_ARRAY_JOB_ID": "424242",
            "SLURM_ARRAY_TASK_ID": "2",
            "CONFIG_DIR": str(tmp_path / "unused-config-dir"),
            "PYTHON_BIN": str(stub),
            "SRUN_BIN": "missing-srun",
            "STUB_LOG": str(tmp_path / "stub_log.txt"),
        },
    )

    assert result.returncode == 0
    published = tmp_path / "results/config-manifest-424242.txt"
    assert published.read_text() == f"{first}\n{second}\n"
    assert not (tmp_path / "results/.config-manifest-424242.lock").exists()
    assert not list((tmp_path / "results").glob(".config-manifest.*"))
    log = (tmp_path / "stub_log.txt").read_text()
    assert log.count("\n") == 2, f"expected generation plus one run, got: {log}"
    assert f"--config {second}" in log


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
