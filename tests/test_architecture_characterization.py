import torch
from src.architectures.cafo_cnn import CaFo_CNN
from src.architectures.mf_mlp import MF_MLP
from torch import nn

from beyond_backprop.architectures import build_fair_bp_baseline


def test_mf_bp_baseline_has_native_dimensions_without_projection_matrices() -> None:
    config = {
        "algorithm": {"name": "BP"},
        "model": {"name": "MF_MLP", "params": {"hidden_dims": [5, 4]}},
        "data": {
            "name": "MNIST",
            "input_channels": 1,
            "image_size": 4,
            "num_classes": 3,
        },
    }
    baseline = build_fair_bp_baseline(config)
    native = MF_MLP(input_dim=16, hidden_dims=[5, 4], num_classes=3)

    assert isinstance(baseline, nn.Sequential)
    assert not any("projection" in name for name, _ in baseline.named_parameters())
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel()
        for name, parameter in native.named_parameters()
        if "projection_matrices" not in name
    )
    assert baseline(torch.randn(2, 16)).shape == (2, 3)


def test_cafo_shape_probe_does_not_mutate_batchnorm_state() -> None:
    model = CaFo_CNN(
        input_channels=1,
        block_channels=[4, 8],
        image_size=8,
        num_classes=3,
        use_batchnorm=True,
    )

    for block in model.blocks:
        assert isinstance(block.bn, nn.BatchNorm2d)
        assert torch.equal(
            block.bn.running_mean, torch.zeros_like(block.bn.running_mean)
        )
        assert torch.equal(block.bn.running_var, torch.ones_like(block.bn.running_var))
        assert int(block.bn.num_batches_tracked) == 0


def test_cafo_bp_baseline_is_trainable_and_has_classifier_output() -> None:
    config = {
        "algorithm": {"name": "BP"},
        "model": {
            "name": "CaFo_CNN",
            "params": {"block_channels": [4, 8], "use_batchnorm": True},
        },
        "data": {"input_channels": 1, "image_size": 8, "num_classes": 3},
    }
    baseline = build_fair_bp_baseline(config)

    assert baseline.training
    assert baseline(torch.randn(2, 1, 8, 8)).shape == (2, 3)
