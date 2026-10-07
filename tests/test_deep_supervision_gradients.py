"""Proves the ablation ladder can distinguish joint gradients from local ones.

The BP-DS and MF-Joint rungs exist to separate "auxiliary supervision" from
"forward-only locality". That separation is only real if a loss attached to the
last hidden layer can still reach the first layer's weights. If
``forward_with_intermediate_activations`` detached internally, BP-DS would
silently collapse onto MF and the ladder would measure nothing.
"""

import unittest

import torch
import torch.nn as nn

from src.algorithms.mf import mf_local_loss_fn
from src.architectures.mf_mlp import MF_MLP

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


def _first_linear(model: MF_MLP) -> nn.Linear:
    return model.layers[0]


class TestDeepSupervisionGradientFlow(unittest.TestCase):
    """Task 1 of Phase 3: the gradient test that gates the whole ladder."""

    def test_deepest_auxiliary_loss_reaches_layer_zero(self) -> None:
        """The layer-L auxiliary loss must produce a non-zero grad on W_1."""
        model = _build_model()
        inputs, targets = _batch()

        activations = model.forward_with_intermediate_activations(inputs)
        self.assertEqual(len(activations), len(HIDDEN_DIMS) + 1)

        deepest = len(HIDDEN_DIMS)
        loss = mf_local_loss_fn(
            activations[deepest],
            model.get_projection_matrix(deepest),
            targets,
            nn.CrossEntropyLoss(),
        )
        model.zero_grad()
        loss.backward()

        grad = _first_linear(model).weight.grad
        self.assertIsNotNone(
            grad,
            "W_1 received no gradient from the layer-L auxiliary loss: "
            "forward_with_intermediate_activations detaches, so BP-DS would "
            "collapse onto MF and the ablation ladder is void.",
        )
        self.assertGreater(
            float(grad.abs().sum()),
            0.0,
            "W_1's gradient from the layer-L auxiliary loss is identically zero.",
        )

    def test_every_intermediate_activation_is_attached(self) -> None:
        """Each a_i for i>0 must carry grad history back to the input graph."""
        model = _build_model()
        inputs, _ = _batch()
        inputs.requires_grad_(True)

        activations = model.forward_with_intermediate_activations(inputs)
        for index, activation in enumerate(activations[1:], start=1):
            self.assertTrue(
                activation.requires_grad,
                f"a_{index} is detached from the autograd graph.",
            )
            self.assertIsNotNone(
                activation.grad_fn, f"a_{index} has no grad_fn."
            )

    def test_shallow_auxiliary_loss_leaves_deeper_layers_untouched(self) -> None:
        """A loss at a_i must not reach W_j for j>i, or the ladder is confounded."""
        model = _build_model()
        inputs, targets = _batch()

        activations = model.forward_with_intermediate_activations(inputs)
        loss = mf_local_loss_fn(
            activations[1],
            model.get_projection_matrix(1),
            targets,
            nn.CrossEntropyLoss(),
        )
        model.zero_grad()
        loss.backward()

        self.assertGreater(float(_first_linear(model).weight.grad.abs().sum()), 0.0)
        for deeper in range(1, len(HIDDEN_DIMS)):
            self.assertIsNone(
                model.layers[deeper * 2].weight.grad,
                f"W_{deeper + 1} received a gradient from the a_1 auxiliary loss.",
            )

    def test_detachment_is_what_makes_mf_local(self) -> None:
        """The MF training loop's ``.detach()`` is the only thing blocking flow.

        Guards the ladder's single-variable claim: rungs 3 and 4 differ by this
        call alone, so removing it must be sufficient to restore joint gradients.
        """
        model = _build_model()
        inputs, targets = _batch()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            prefix = model.layers[1](model.layers[0](inputs))
        detached_activation = model.layers[3](model.layers[2](prefix.detach()))
        loss = mf_local_loss_fn(
            detached_activation, model.get_projection_matrix(2), targets, criterion
        )
        model.zero_grad()
        loss.backward()
        self.assertIsNone(
            _first_linear(model).weight.grad,
            "MF's detach failed to isolate W_1; MF is not layer-local.",
        )

        activations = model.forward_with_intermediate_activations(inputs)
        joint_loss = mf_local_loss_fn(
            activations[2], model.get_projection_matrix(2), targets, criterion
        )
        model.zero_grad()
        joint_loss.backward()
        self.assertGreater(
            float(_first_linear(model).weight.grad.abs().sum()),
            0.0,
            "Without detach, W_1 still received no gradient.",
        )

        self.assertAlmostEqual(
            float(loss.item()), float(joint_loss.item()), places=6,
            msg="Detaching changed the forward value, so the rungs differ by "
            "more than gradient flow.",
        )


if __name__ == "__main__":
    unittest.main()
