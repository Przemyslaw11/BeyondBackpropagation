from types import SimpleNamespace
from unittest.mock import patch

import torch
from src.training import engine


def test_run_training_forwards_download_policy() -> None:
    config = {
        "general": {"backend": "local"},
        "data": {"name": "MNIST", "download": False},
        "data_loader": {"batch_size": 2},
    }
    with (
        patch.object(
            engine,
            "_setup_environment_and_wandb",
            return_value=(7, torch.device("cpu"), None),
        ),
        patch.object(
            engine,
            "_setup_hardware_monitors",
            return_value=(False, None, None, None, None, {}),
        ),
        patch.object(engine, "_finalize_run"),
        patch.object(
            engine, "get_dataloaders", side_effect=RuntimeError("stop")
        ) as loader,
    ):
        result = engine.run_training(config)

    assert "error" in result
    assert loader.call_args.kwargs["download"] is False


def test_run_training_finalizes_when_setup_fails_before_monitor_creation() -> None:
    with (
        patch.object(
            engine, "_setup_environment_and_wandb", side_effect=RuntimeError("stop")
        ),
        patch.object(engine, "_finalize_run") as finalize,
    ):
        result = engine.run_training({})

    assert result["error"] == "stop"
    args = finalize.call_args.args
    assert args[3] is None  # tracker
    assert args[4] is None  # CodeCarbon CSV path
    assert args[5] is None  # profiler/energy monitor
    assert args[6] is False  # NVML active
    assert args[7] is None  # NVML handle


def test_legacy_runner_removes_the_actual_codecarbon_result_key() -> None:
    from scripts import run_experiment

    results = {"codecarbon_emissions_gCO2e": 1.0, "test_accuracy": 50.0}
    config = {"experiment_name": "test", "general": {"backend": "local"}}
    args = SimpleNamespace(config="unused.yaml", backend=None)

    with (
        patch.object(run_experiment, "load_config", return_value=config),
        patch.object(run_experiment, "get_execution_backend") as backend_factory,
        patch.object(run_experiment, "create_directory_if_not_exists"),
        patch.object(run_experiment, "setup_logging"),
        patch.object(run_experiment, "run_training", return_value=results),
    ):
        backend = backend_factory.return_value
        backend.resolve_log_file.return_value = "results/test.log"
        run_experiment.main(args)

    assert "codecarbon_emissions_gCO2e" not in results
    assert results["test_accuracy"] == 50.0
