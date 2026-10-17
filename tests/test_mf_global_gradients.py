"""Guards the Seq-Global rung (W1 attribution ladder, Phase-2 unit tests).

Seq-Global shares MF-Seq's stopping schedule and local readout but unfreezes the
prefix: at stage s the only parameters that may move are {W_1..W_s, M_s}. If a
stage loss could not reach W_1, or if frozen/deeper parameters moved anyway, the
rung would not isolate "backprop through the trainable prefix" and the ladder's
schedule-vs-locality separator would be void.
"""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.mf import mf_local_loss_fn
from src.algorithms.mf_global import train_mf_global_model
from src.architectures.mf_mlp import MF_MLP

BATCH_SIZE = 8
NUM_SAMPLES = 32
INPUT_DIM = 12
HIDDEN_DIMS = [10, 9, 7]
NUM_CLASSES = 5


def _build_model() -> MF_MLP:
    torch.manual_seed(0)
    return MF_MLP(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        activation="relu",
    )


def _batch() -> tuple:
    torch.manual_seed(1)
    inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return inputs, targets


def _tiny_loaders() -> tuple:
    torch.manual_seed(2)
    inputs = torch.randn(NUM_SAMPLES, INPUT_DIM)
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def _train_config(train_m0: bool) -> dict:
    return {
        "algorithm_params": {
            "optimizer_type": "Adam",
            "lr": 0.01,
            "weight_decay": 0.0,
            "log_interval": 9999,
            "activation_cache": "recompute",
            "mf_train_m0": train_m0,
        },
        "early_stopping": {
            "enabled": True,
            "metric": "val_loss",
            "mode": "min",
            "min_delta": 0.0,
            "patience": 2,
            "max_epochs": 3,
        },
    }


class TestSeqGlobalGradientReach(unittest.TestCase):
    def test_stage_loss_reaches_full_trainable_prefix(self) -> None:
        """At stage s=2, a_2's local loss must reach W_1 and W_2, not W_3."""
        model = _build_model()
        inputs, targets = _batch()

        for p in model.parameters():
            p.requires_grad_(False)
        for k in range(2):
            for p in model.layers[k * 2].parameters():
                p.requires_grad_(True)
        projection = model.get_projection_matrix(2)
        projection.requires_grad_(True)

        activations = model.forward_with_intermediate_activations(inputs)
        loss = mf_local_loss_fn(
            activations[2], projection, targets, nn.CrossEntropyLoss()
        )
        model.zero_grad()
        loss.backward()

        for k in range(2):
            grad = model.layers[k * 2].weight.grad
            self.assertIsNotNone(grad, f"W_{k + 1} received no gradient at stage 2.")
            self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertIsNotNone(projection.grad)
        self.assertGreater(float(projection.grad.abs().sum()), 0.0)

        self.assertIsNone(
            model.layers[4].weight.grad,
            "W_3 received a gradient from stage 2's local loss.",
        )
        for m_index in (0, 1, 3):
            self.assertIsNone(
                model.get_projection_matrix(m_index).grad,
                f"M_{m_index} received a gradient from stage 2's local loss.",
            )

    def test_one_stage_step_moves_prefix_and_its_readout_only(self) -> None:
        """Optimizer step at stage 2 must leave W_3 and every other M_j bitwise."""
        model = _build_model()
        inputs, targets = _batch()

        for p in model.parameters():
            p.requires_grad_(False)
        params = []
        for k in range(2):
            for p in model.layers[k * 2].parameters():
                p.requires_grad_(True)
            params.extend(model.layers[k * 2].parameters())
        projection = model.get_projection_matrix(2)
        projection.requires_grad_(True)
        params.append(projection)

        before = {
            "w1": model.layers[0].weight.detach().clone(),
            "w2": model.layers[2].weight.detach().clone(),
            "w3": model.layers[4].weight.detach().clone(),
        }
        before.update(
            {f"m{m}": model.get_projection_matrix(m).detach().clone() for m in (0, 1, 2, 3)}
        )

        optimizer = torch.optim.Adam(params, lr=0.01)
        activations = model.forward_with_intermediate_activations(inputs)
        loss = mf_local_loss_fn(
            activations[2], projection, targets, nn.CrossEntropyLoss()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertFalse(
            torch.equal(model.layers[0].weight.detach(), before["w1"]),
            "W_1 did not move during a stage-2 step.",
        )
        self.assertFalse(
            torch.equal(model.layers[2].weight.detach(), before["w2"]),
            "W_2 did not move during a stage-2 step.",
        )
        self.assertTrue(
            torch.equal(model.layers[4].weight.detach(), before["w3"]),
            "W_3 moved during a stage-2 step.",
        )
        self.assertFalse(
            torch.equal(model.get_projection_matrix(2).detach(), before["m2"]),
            "M_2 did not move during a stage-2 step.",
        )
        for m in (0, 1, 3):
            self.assertTrue(
                torch.equal(model.get_projection_matrix(m).detach(), before[f"m{m}"]),
                f"M_{m} moved during a stage-2 step.",
            )

    def test_activation_cache_is_rejected(self) -> None:
        model = _build_model()
        train_loader, val_loader = _tiny_loaders()
        config = _train_config(train_m0=False)
        config["algorithm_params"]["activation_cache"] = "cache_device"
        with self.assertRaises(ValueError):
            train_mf_global_model(
                model,
                train_loader,
                config,
                torch.device("cpu"),
                input_adapter=lambda x: x,
                val_loader=val_loader,
            )


class TestSeqGlobalTrainingLoop(unittest.TestCase):
    def test_m0_off_skips_m0_and_counts_stage_epochs(self) -> None:
        model = _build_model()
        train_loader, val_loader = _tiny_loaders()
        m0_before = model.get_projection_matrix(0).detach().clone()

        config = _train_config(train_m0=False)
        train_mf_global_model(
            model,
            train_loader,
            config,
            torch.device("cpu"),
            input_adapter=lambda x: x,
            val_loader=val_loader,
        )

        self.assertTrue(
            torch.equal(model.get_projection_matrix(0).detach(), m0_before),
            "M_0 moved although mf_train_m0 is False.",
        )
        epochs = config["_run_stats"]["epochs_completed"]
        self.assertGreater(epochs, 0)
        self.assertLessEqual(epochs, len(HIDDEN_DIMS) * 3)

    def test_m0_on_trains_m0_and_adds_its_epochs(self) -> None:
        model = _build_model()
        train_loader, val_loader = _tiny_loaders()
        m0_before = model.get_projection_matrix(0).detach().clone()

        config = _train_config(train_m0=True)
        train_mf_global_model(
            model,
            train_loader,
            config,
            torch.device("cpu"),
            input_adapter=lambda x: x,
            val_loader=val_loader,
        )

        self.assertFalse(
            torch.equal(model.get_projection_matrix(0).detach(), m0_before),
            "M_0 did not move although mf_train_m0 is True.",
        )
        self.assertGreaterEqual(
            config["_run_stats"]["epochs_completed"], 1
        )


if __name__ == "__main__":
    unittest.main()
