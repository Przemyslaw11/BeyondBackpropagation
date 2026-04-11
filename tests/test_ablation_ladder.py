"""Guards the single-variable steps between the rungs of the ablation ladder.

The ladder only decomposes BP into MF if each rung differs from its neighbour in
exactly one respect. These tests pin the three that are easy to break silently:
which heads carry an auxiliary loss, which parameters an optimiser may touch, and
whether an activation cache can leak onto a joint-gradient rung.
"""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms import get_evaluation_function, get_training_function
from src.algorithms.mf import mf_local_loss_fn
from src.architectures.mf_mlp import MF_MLP
from src.baselines.bp_ds import (
    READOUT_OUTPUT_LAYER,
    READOUT_PROJECTION,
    _auxiliary_indices,
    _deep_supervised_losses,
    _select_trainable_parameters,
    evaluate_bp_ds_model,
    evaluate_mf_joint_model,
    train_bp_ds_model,
    train_mf_joint_model,
)

INPUT_DIM = 12
HIDDEN_DIMS = [10, 9, 7]
NUM_CLASSES = 5
BATCH_SIZE = 8


def _model() -> MF_MLP:
    torch.manual_seed(0)
    return MF_MLP(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        activation="relu",
    )


def _batch():
    torch.manual_seed(1)
    return (
        torch.randn(BATCH_SIZE, INPUT_DIM),
        torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)),
    )


def _loader() -> DataLoader:
    inputs, targets = _batch()
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


def _config(algorithm: str, **algorithm_params) -> dict:
    return {
        "algorithm": {"name": algorithm},
        "algorithm_params": algorithm_params,
        "optimizer": {"lr": 0.01},
        "early_stopping": {"enabled": False, "max_epochs": 1},
        "training": {"log_interval": 9999},
    }


class TestAuxiliaryHeadPlacement(unittest.TestCase):
    """Rung 2 supervises every M_i; rung 3 promotes M_L to the readout."""

    def test_bp_ds_supervises_every_projection_matrix(self) -> None:
        model = _model()
        self.assertEqual(
            _auxiliary_indices(model, READOUT_OUTPUT_LAYER),
            list(range(len(HIDDEN_DIMS) + 1)),
        )

    def test_mf_joint_excludes_the_readout_matrix_from_the_auxiliary_set(self) -> None:
        model = _model()
        self.assertEqual(
            _auxiliary_indices(model, READOUT_PROJECTION), list(range(len(HIDDEN_DIMS)))
        )

    def test_mf_joint_objective_equals_mf_sum_of_local_losses(self) -> None:
        """At aux_weight 1.0 rung 3 must differ from rung 4 by detachment alone."""
        model = _model()
        inputs, targets = _batch()
        criterion = nn.CrossEntropyLoss()

        activations = model.forward_with_intermediate_activations(inputs)
        total, _, _ = _deep_supervised_losses(
            model, activations, targets, criterion, READOUT_PROJECTION, 1.0
        )

        mf_objective = sum(
            mf_local_loss_fn(
                activations[i], model.get_projection_matrix(i), targets, criterion
            )
            for i in range(len(HIDDEN_DIMS) + 1)
        )
        self.assertAlmostEqual(float(total), float(mf_objective), places=6)

    def test_aux_weight_scales_only_the_auxiliary_terms(self) -> None:
        model = _model()
        inputs, targets = _batch()
        criterion = nn.CrossEntropyLoss()
        activations = model.forward_with_intermediate_activations(inputs)

        readout_only, _, aux_at_zero = _deep_supervised_losses(
            model, activations, targets, criterion, READOUT_OUTPUT_LAYER, 0.0
        )
        doubled, _, aux_at_two = _deep_supervised_losses(
            model, activations, targets, criterion, READOUT_OUTPUT_LAYER, 2.0
        )
        self.assertAlmostEqual(aux_at_zero, 0.0, places=6)
        self.assertAlmostEqual(
            float(doubled) - float(readout_only), aux_at_two, places=5
        )


class TestParameterOwnership(unittest.TestCase):
    """A head a rung never reads must not collect optimiser state or weight decay."""

    def test_bp_ds_trains_every_parameter(self) -> None:
        model = _model()
        trainable = _select_trainable_parameters(model, READOUT_OUTPUT_LAYER)
        self.assertEqual(len(trainable), len(list(model.parameters())))

    def test_mf_joint_freezes_the_unused_output_layer(self) -> None:
        model = _model()
        _select_trainable_parameters(model, READOUT_PROJECTION)
        for name, param in model.named_parameters():
            expected = not name.startswith("output_layer")
            self.assertEqual(
                param.requires_grad,
                expected,
                f"{name}.requires_grad should be {expected} for MF-Joint.",
            )

    def test_mf_joint_leaves_the_output_layer_untouched_by_training(self) -> None:
        model = _model()
        before = model.output_layer.weight.detach().clone()
        train_mf_joint_model(
            model=model,
            train_loader=_loader(),
            val_loader=None,
            config=_config("mf_joint"),
            device=torch.device("cpu"),
            input_adapter=lambda x: x.view(x.shape[0], -1),
        )
        self.assertTrue(torch.equal(before, model.output_layer.weight.detach()))


class TestActivationCacheIsRejected(unittest.TestCase):
    """Rungs 1-3 propagate joint gradients, so a cached activation goes stale."""

    def _assert_rejected(self, train_fn, algorithm: str) -> None:
        for strategy in ("cache_device", "cache_host"):
            with self.subTest(algorithm=algorithm, strategy=strategy):
                with self.assertRaises(ValueError) as ctx:
                    train_fn(
                        model=_model(),
                        train_loader=_loader(),
                        val_loader=None,
                        config=_config(algorithm, activation_cache=strategy),
                        device=torch.device("cpu"),
                        input_adapter=lambda x: x.view(x.shape[0], -1),
                    )
                self.assertIn("layer-sequential", str(ctx.exception))

    def test_bp_ds_rejects_a_cache_strategy(self) -> None:
        self._assert_rejected(train_bp_ds_model, "bp_ds")

    def test_mf_joint_rejects_a_cache_strategy(self) -> None:
        self._assert_rejected(train_mf_joint_model, "mf_joint")


class TestDispatch(unittest.TestCase):
    """The engine must reach the new rungs by config name alone."""

    def test_training_and_evaluation_dispatch(self) -> None:
        self.assertIs(get_training_function("bp_ds"), train_bp_ds_model)
        self.assertIs(get_training_function("mf_joint"), train_mf_joint_model)
        self.assertIs(get_evaluation_function("bp_ds"), evaluate_bp_ds_model)
        self.assertIs(get_evaluation_function("mf_joint"), evaluate_mf_joint_model)

    def test_evaluation_reads_out_through_different_heads(self) -> None:
        model = _model()
        criterion = nn.CrossEntropyLoss()
        adapter = lambda x: x.view(x.shape[0], -1)  # noqa: E731
        bp_ds = evaluate_bp_ds_model(model, _loader(), criterion, torch.device("cpu"), adapter)
        mf_joint = evaluate_mf_joint_model(
            model, _loader(), criterion, torch.device("cpu"), adapter
        )
        self.assertNotAlmostEqual(bp_ds[0], mf_joint[0], places=4)


class TestJointGradientsReachTheFirstLayer(unittest.TestCase):
    """The property the whole ladder rests on, asserted through the real trainers."""

    def test_one_bp_ds_step_moves_the_first_layer(self) -> None:
        for name, train_fn in (
            ("bp_ds", train_bp_ds_model),
            ("mf_joint", train_mf_joint_model),
        ):
            with self.subTest(algorithm=name):
                model = _model()
                before = model.layers[0].weight.detach().clone()
                train_fn(
                    model=model,
                    train_loader=_loader(),
                    val_loader=None,
                    config=_config(name),
                    device=torch.device("cpu"),
                    input_adapter=lambda x: x.view(x.shape[0], -1),
                )
                self.assertFalse(
                    torch.equal(before, model.layers[0].weight.detach()),
                    f"{name} left W_1 unchanged, so gradients never reached it.",
                )


if __name__ == "__main__":
    unittest.main()
