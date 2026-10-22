"""Guards the post-training diagnostic endpoints added on 2026-10-22.

The endpoints must be pure observation: same seed, same final weights, no
optimiser step, no model mutation.  They also must be outside the NVML energy
window, which the engine enforces by calling them after the training context
manager has closed.
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.mf import train_mf_model
from src.architectures.mf_mlp import MF_MLP
from src.baselines.bp import train_bp_model
from src.training.secondary_endpoints import compute_secondary_endpoints

INPUT_DIM = 12
HIDDEN_DIMS = [10, 9, 7]
NUM_CLASSES = 5
BATCH_SIZE = 32


def _data():
    torch.manual_seed(7)
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return TensorDataset(inputs, targets)


def _loader():
    return DataLoader(_data(), batch_size=8, shuffle=False)


def _bp_model():
    torch.manual_seed(11)
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIMS[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMS[2], NUM_CLASSES),
    )


def _mf_model():
    torch.manual_seed(11)
    return MF_MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, activation="relu")


def _bp_config():
    return {
        "experiment_name": "unit_secondary_bp",
        "algorithm": {"name": "BP"},
        "data": {"name": "MNIST", "num_classes": NUM_CLASSES},
        "training": {"criterion": "CrossEntropyLoss", "log_interval": 9999},
        "optimizer": {"type": "AdamW", "lr": 0.01, "weight_decay": 0.0},
        "early_stopping": {"enabled": False, "max_epochs": 2},
        "checkpointing": {"checkpoint_dir": None},
        "logging": {"wandb": {"use_wandb": False}},
    }


def _mf_config():
    return {
        "experiment_name": "unit_secondary_mf",
        "algorithm": {"name": "MF"},
        "data": {"name": "MNIST", "num_classes": NUM_CLASSES},
        "algorithm_params": {
            "optimizer_type": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "mf_train_m0": True,
            "log_interval": 9999,
        },
        "early_stopping": {"enabled": False, "max_epochs": 1},
        "checkpointing": {"checkpoint_dir": None},
        "logging": {"wandb": {"use_wandb": False}},
    }


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_same_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"state changed at {key}"


def test_secondary_endpoints_preserve_bp_training_and_weights():
    loader = _loader()
    adapter = lambda x: x  # noqa: E731

    torch.manual_seed(123)
    model_a = _bp_model()
    cfg_a = _bp_config()
    train_bp_model(
        model_a,
        train_loader=loader,
        val_loader=loader,
        config=cfg_a,
        device=torch.device("cpu"),
        input_adapter=adapter,
    )

    torch.manual_seed(123)
    model_b = _bp_model()
    cfg_b = _bp_config()
    train_bp_model(
        model_b,
        train_loader=loader,
        val_loader=loader,
        config=cfg_b,
        device=torch.device("cpu"),
        input_adapter=adapter,
    )

    before = _state(model_b)
    endpoints = compute_secondary_endpoints(
        model=model_b,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        config=cfg_b,
        device=torch.device("cpu"),
        algo_name="bp",
        input_adapter=adapter,
        test_loss=0.0,
        test_accuracy=50.0,
    )
    _assert_same_state(before, _state(model_b))
    _assert_same_state(_state(model_a), _state(model_b))

    assert endpoints["final_train_accuracy"] == endpoints["final_train_accuracy"]
    assert "a_1" in endpoints["linear_probe_test_accuracy"]
    assert "a_3" in endpoints["effective_rank"]
    assert endpoints["local_head_test_accuracy"] == {}
    assert endpoints["secondary_eval_seconds"] >= 0.0
    assert cfg_b["_run_stats"]["stage_epochs"][0]["stage"] == "backprop"
    assert cfg_b["_run_stats"]["stage_epochs"][0]["stopping_epoch"] == 2
    assert cfg_b["_run_stats"]["epochs_completed"] == 2


def test_secondary_endpoints_support_mf_and_heads():
    loader = _loader()
    model = _mf_model()
    cfg = _mf_config()
    before = copy.deepcopy(_state(model))
    endpoints = compute_secondary_endpoints(
        model=model,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        config=cfg,
        device=torch.device("cpu"),
        algo_name="mf",
        input_adapter=lambda x: x,
        test_loss=0.0,
        test_accuracy=50.0,
    )
    _assert_same_state(before, _state(model))
    assert set(endpoints["linear_probe_test_accuracy"]) == {"a_1", "a_2", "a_3"}
    assert set(endpoints["local_head_test_accuracy"]) == {
        "M_0",
        "M_1",
        "M_2",
        "M_3",
    }


def test_mf_stage_epochs_are_recorded():
    loader = _loader()
    model = _mf_model()
    cfg = _mf_config()
    train_mf_model(
        model,
        train_loader=loader,
        config=cfg,
        device=torch.device("cpu"),
        input_adapter=lambda x: x,
        val_loader=loader,
    )
    stages = cfg["_run_stats"]["stage_epochs"]
    assert [stage["stage"] for stage in stages] == [
        "M0",
        "Layer_W1_M1",
        "Layer_W2_M2",
        "Layer_W3_M3",
    ]
    assert cfg["_run_stats"]["epochs_completed"] == 4
    assert all(stage["stopping_epoch"] == 1 for stage in stages)


def test_secondary_endpoints_preserve_mf_training_and_weights():
    """MF training plus the post-training probe must equal a clean MF run."""
    loader = _loader()

    torch.manual_seed(123)
    model_a = _mf_model()
    cfg_a = _mf_config()
    train_mf_model(
        model_a,
        train_loader=loader,
        config=cfg_a,
        device=torch.device("cpu"),
        input_adapter=lambda x: x,
        val_loader=loader,
    )

    torch.manual_seed(123)
    model_b = _mf_model()
    cfg_b = _mf_config()
    train_mf_model(
        model_b,
        train_loader=loader,
        config=cfg_b,
        device=torch.device("cpu"),
        input_adapter=lambda x: x,
        val_loader=loader,
    )

    before = _state(model_b)
    endpoints = compute_secondary_endpoints(
        model=model_b,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        config=cfg_b,
        device=torch.device("cpu"),
        algo_name="mf",
        input_adapter=lambda x: x,
        test_loss=0.0,
        test_accuracy=50.0,
    )
    _assert_same_state(before, _state(model_b))
    _assert_same_state(_state(model_a), _state(model_b))
    assert set(endpoints["local_head_test_accuracy"]) == {
        "M_0",
        "M_1",
        "M_2",
        "M_3",
    }
