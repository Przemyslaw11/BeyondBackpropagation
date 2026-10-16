"""Guards the MF-Simul rung: detachment without staging (Phase 1 unit tests).

MF-Simul changes MF-Joint in exactly one way: every hidden layer consumes a
detached copy of its input. That claim has three testable halves:

1. forward values (hence the loss value) must be unchanged by the detach;
2. each local loss must reach exactly its own layer's parameters;
3. one joint MF-Simul step must move M0 and every (W_i, M_i) exactly as one
   step of MF-Seq's per-stage objectives would, because that equivalence is what
   licenses calling the schedule the single difference between the rungs.
"""

import copy
import unittest

import torch
import torch.nn as nn

from src.algorithms.mf import mf_local_loss_fn
from src.architectures.mf_mlp import MF_MLP
from src.baselines.bp_ds import (
    READOUT_PROJECTION,
    _deep_supervised_losses,
    _detached_intermediate_activations,
    _select_trainable_parameters,
)

BATCH_SIZE = 8
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


class TestMfSimulDetachment(unittest.TestCase):
    """The rung is only a single-variable control if all three halves hold."""

    def test_forward_values_unchanged_by_detachment(self) -> None:
        """Detaching inputs must not perturb any activation or the total loss."""
        model = _build_model()
        inputs, targets = _batch()
        criterion = nn.CrossEntropyLoss()

        plain = model.forward_with_intermediate_activations(inputs)
        detached = _detached_intermediate_activations(model, inputs)
        self.assertEqual(len(plain), len(detached))
        for index, (a_plain, a_detached) in enumerate(zip(plain, detached)):
            self.assertTrue(
                torch.allclose(a_plain, a_detached),
                f"a_{index} changed when its input was detached.",
            )

        loss_plain, _, _ = _deep_supervised_losses(
            model, plain, targets, criterion, READOUT_PROJECTION, 1.0
        )
        loss_detached, _, _ = _deep_supervised_losses(
            model, detached, targets, criterion, READOUT_PROJECTION, 1.0
        )
        self.assertAlmostEqual(
            loss_plain.detach().item(), loss_detached.detach().item(), places=6
        )

    def test_each_local_loss_reaches_only_its_own_layer(self) -> None:
        """Loss at a_i may graze M_i and W_i only, never any other layer."""
        model = _build_model()
        inputs, targets = _batch()
        criterion = nn.CrossEntropyLoss()
        activations = _detached_intermediate_activations(model, inputs)
        last = len(HIDDEN_DIMS)

        for index in range(last + 1):
            model.zero_grad()
            loss = mf_local_loss_fn(
                activations[index],
                model.get_projection_matrix(index),
                targets,
                criterion,
            )
            loss.backward()

            own_matrix = model.get_projection_matrix(index).grad
            self.assertIsNotNone(own_matrix, f"M_{index} got no gradient.")
            self.assertGreater(float(own_matrix.abs().sum()), 0.0)
            for other in range(last + 1):
                if other == index:
                    continue
                self.assertIsNone(
                    model.get_projection_matrix(other).grad,
                    f"loss at a_{index} reached M_{other}.",
                )

            for w in range(last):
                grad = model.layers[w * 2].weight.grad
                if index >= 1 and w == index - 1:
                    self.assertIsNotNone(grad, f"W_{w + 1} got no gradient from a_{index}.")
                    self.assertGreater(float(grad.abs().sum()), 0.0)
                else:
                    self.assertIsNone(
                        grad,
                        f"loss at a_{index} reached W_{w + 1}; detachment is broken.",
                    )

    def test_one_joint_step_matches_staged_local_steps(self) -> None:
        """One MF-Simul step equals one step of each MF-Seq unit's objective."""
        criterion = nn.CrossEntropyLoss()
        lr = 0.01
        model_simul = _build_model()
        model_ref = copy.deepcopy(model_simul)
        initial_state = {
            name: param.detach().clone()
            for name, param in model_simul.named_parameters()
        }
        inputs, targets = _batch()
        last = len(HIDDEN_DIMS)

        optimizer = torch.optim.Adam(
            _select_trainable_parameters(model_simul, READOUT_PROJECTION), lr=lr
        )
        activations = _detached_intermediate_activations(model_simul, inputs)
        loss, _, _ = _deep_supervised_losses(
            model_simul, activations, targets, criterion, READOUT_PROJECTION, 1.0
        )
        model_simul.zero_grad()
        loss.backward()
        optimizer.step()

        reference_activations = _detached_intermediate_activations(model_ref, inputs)
        units = []
        for index in range(last + 1):
            if index == 0:
                params = [model_ref.get_projection_matrix(0)]
                unit_loss = mf_local_loss_fn(
                    inputs, model_ref.get_projection_matrix(0), targets, criterion
                )
            else:
                params = list(model_ref.layers[(index - 1) * 2].parameters()) + [
                    model_ref.get_projection_matrix(index)
                ]
                unit_loss = mf_local_loss_fn(
                    reference_activations[index],
                    model_ref.get_projection_matrix(index),
                    targets,
                    criterion,
                )
            units.append((params, unit_loss))

        for params, unit_loss in units:
            unit_optimizer = torch.optim.Adam(params, lr=lr)
            model_ref.zero_grad()
            unit_loss.backward()
            unit_optimizer.step()

        simul_state = dict(model_simul.named_parameters())
        ref_state = dict(model_ref.named_parameters())
        moved = 0
        for name, param in ref_state.items():
            if name.startswith("output_layer"):
                continue
            self.assertTrue(
                torch.allclose(simul_state[name], param, atol=1e-7, rtol=0.0),
                f"{name} diverges between one joint MF-Simul step and the "
                "staged local steps.",
            )
            if not torch.allclose(initial_state[name], param):
                moved += 1
        self.assertGreater(moved, 0, "Nothing moved; the test compared no-ops.")


if __name__ == "__main__":
    unittest.main()
