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
