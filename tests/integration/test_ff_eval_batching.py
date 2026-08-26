"""P4 equivalence tests: stacked-candidate FF inference vs the candidate loop.

The production ``evaluate_ff_model`` evaluates all label candidates in one
stacked forward pass (review P4). These tests pin bit-equivalence against the
verbatim per-candidate loop it replaced, across seeds and shapes, satisfying
the review's adoption bar (bit-identical predictions on a fixed seed).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms.ff import evaluate_ff_model
from beyond_backprop.algorithms.ff_math import (
    aggregate_goodness,
    generate_hinton_inputs,
)
from beyond_backprop.architectures.ff_mlp import FF_MLP

DEVICE = torch.device("cpu")


def _reference_batch_goodness(
    model: FF_MLP, images: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Verbatim pre-P4 per-candidate evaluation for one batch."""
    batch_size = images.shape[0]
    goodness = torch.zeros((batch_size, num_classes))
    for label_candidate in range(num_classes):
        candidate_labels = torch.full((batch_size,), label_candidate, dtype=torch.long)
        ff_input, _, _ = generate_hinton_inputs(
            images, candidate_labels, num_classes, DEVICE
        )
        goodness[:, label_candidate] = aggregate_goodness(
            model.forward_goodness_per_layer(ff_input)
        )
    return goodness


def _config(image_size: int, hidden: list[int], classes: int) -> dict:
    return {
        "model": {"name": "FF_MLP", "params": {"hidden_dims": hidden}},
        "data": {"input_channels": 1, "image_size": image_size, "num_classes": classes},
        "algorithm_params": {},
        "data_loader": {},
    }


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    ["image_size", "hidden", "classes"],
    [(2, [3], 2), (5, [8], 3), (14, [32, 32], 10)],
)
def test_stacked_candidates_bitwise_match_candidate_loop(
    seed: int, image_size: int, hidden: list[int], classes: int
) -> None:
    torch.manual_seed(seed)
    model = FF_MLP(_config(image_size, hidden, classes), DEVICE)
    model.eval()
    images = torch.rand(16, 1, image_size, image_size)

    loop_goodness = _reference_batch_goodness(model, images, classes)

    stacked_images = images.repeat_interleave(classes, dim=0)
    candidate_labels = torch.arange(classes).repeat(images.shape[0])
    stacked_input, _, _ = generate_hinton_inputs(
        stacked_images, candidate_labels, classes, DEVICE
    )
    stacked_goodness = aggregate_goodness(
        model.forward_goodness_per_layer(stacked_input)
    ).reshape(images.shape[0], classes)

    assert torch.equal(stacked_goodness, loop_goodness)
    assert torch.equal(stacked_goodness.argmax(1), loop_goodness.argmax(1))


def test_evaluate_ff_model_accuracy_matches_reference_loop() -> None:
    """End-to-end: the stacked implementation reproduces the loop's accuracy."""
    torch.manual_seed(7)
    model = FF_MLP(_config(5, [8], 3), DEVICE)
    model.eval()
    images = torch.rand(64, 1, 5, 5)
    labels = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    total_correct = 0
    for batch_images, batch_labels in loader:
        goodness = _reference_batch_goodness(model, batch_images, 3)
        total_correct += int((goodness.argmax(1) == batch_labels).sum())
    expected_accuracy = (total_correct / len(images)) * 100.0

    results = evaluate_ff_model(model, loader, DEVICE)
    assert results["eval_accuracy"] == expected_accuracy
