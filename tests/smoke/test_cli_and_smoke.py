import pytest

from beyond_backprop.cli.main import main


@pytest.mark.smoke
def test_validate_config_cli_does_not_load_data(capsys) -> None:
    assert (
        main(["validate-config", "--config", "configs/mf/mnist_mlp_2x1000.yaml"]) == 0
    )
    output = capsys.readouterr().out
    assert "Valid configuration" in output
    assert "config_hash:" in output


def test_inspect_config_cli_reports_resolved_execution_details(capsys) -> None:
    assert (
        main(["inspect-config", "--config", "configs/ff/mnist_mlp_3x1000_ADAMW.yaml"])
        == 0
    )
    output = capsys.readouterr().out
    assert "algorithm: ff" in output
    assert "architecture: ff_mlp" in output
    assert "dataset: mnist" in output


def test_cli_override_and_experiment_dry_run_do_not_load_data(capsys) -> None:
    assert (
        main(
            [
                "experiment",
                "run",
                "--config",
                "configs/mf/mnist_mlp_2x1000.yaml",
                "--set",
                "general.device=cpu",
                "--set",
                "data.download=false",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "dry_run: true" in output
    assert "device: cpu" in output


def test_tune_and_batch_dry_runs_validate_without_starting_work(
    tmp_path, capsys
) -> None:
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        """
experiment_name: tiny
general: {seed: 1, device: cpu, backend: local}
algorithm: {name: bp}
model: {name: MF_MLP, params: {hidden_dims: [2]}}
data: {name: mnist, root: /tmp/offline, download: false, val_split: 0.2, num_classes: 2, input_channels: 1, image_size: 2}
data_loader: {batch_size: 2, num_workers: 0, pin_memory: false}
training: {epochs: 1}
optimizer: {type: AdamW, lr: 0.01, weight_decay: 0.0}
""",
        encoding="utf-8",
    )
    assert main(["tune", "--config", str(config_path), "--dry-run"]) == 0
    assert (
        main(
            [
                "batch",
                "--config-dir",
                str(tmp_path),
                "--glob",
                "*.yaml",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "tuning: validated; no study started" in output
    assert f"validated: {config_path}" in output
