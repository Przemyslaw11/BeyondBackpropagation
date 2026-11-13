"""W0b evidence: the M0 stage is dead compute.

Phase 1 of MF trains M_0 on the raw input, but neither phase 2 (which consumes
a_i for i >= 0 via the frozen prefix) nor evaluation (a_L, M_L) ever reads M_0.
The pass-count multiplier and the energy bill, however, include it. These tests
pin the "dead" half of that claim at unit level; w0b_m0_energy.py quantifies the
energy cost on the canonical runs.
"""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.mf import evaluate_mf_model, train_mf_model
from src.architectures.mf_mlp import MF_MLP

INPUT_DIM = 12
HIDDEN_DIMS = [10, 9, 7]
NUM_CLASSES = 5
NUM_SAMPLES = 24


def _build_model() -> MF_MLP:
    torch.manual_seed(0)
    return MF_MLP(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        activation="relu",
    )


def _loader() -> DataLoader:
    torch.manual_seed(1)
    inputs = torch.randn(NUM_SAMPLES, INPUT_DIM)
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=False)


class TestM0IsDeadCompute(unittest.TestCase):
    def test_evaluation_is_bitwise_independent_of_m0(self) -> None:
        model = _build_model()
        loader = _loader()
        before = evaluate_mf_model(model, loader, torch.device("cpu"), lambda x: x)

        with torch.no_grad():
            model.projection_matrices[0].zero_()
        after = evaluate_mf_model(model, loader, torch.device("cpu"), lambda x: x)

        self.assertEqual(before["eval_accuracy"], after["eval_accuracy"])
        self.assertEqual(before["eval_loss"], after["eval_loss"])

    def test_mf_train_m0_false_leaves_m0_untouched(self) -> None:
        model = _build_model()
        loader = _loader()
        m0_before = model.projection_matrices[0].detach().clone()
        config = {
            "algorithm_params": {
                "optimizer_type": "Adam",
                "lr": 0.01,
                "weight_decay": 0.0,
                "log_interval": 9999,
                "activation_cache": "recompute",
                "mf_train_m0": False,
            },
            "early_stopping": {
                "enabled": True,
                "metric": "val_loss",
                "mode": "min",
                "min_delta": 0.0,
                "patience": 2,
                "max_epochs": 2,
            },
        }
        train_mf_model(
            model,
            loader,
            config,
            torch.device("cpu"),
            input_adapter=lambda x: x,
            val_loader=loader,
        )
        self.assertTrue(
            torch.equal(model.projection_matrices[0].detach(), m0_before),
            "M_0 moved although mf_train_m0 is False.",
        )
        self.assertEqual(config["_run_stats"]["epochs_completed"], len(HIDDEN_DIMS) * 2)


if __name__ == "__main__":
    unittest.main()
