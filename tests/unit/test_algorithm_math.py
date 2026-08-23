from __future__ import annotations

import pytest
import torch

from beyond_backprop.algorithms.cafo_math import aggregate_predictor_outputs
from beyond_backprop.algorithms.ff_math import (
    aggregate_goodness,
    generate_hinton_inputs,
    linear_cooldown_lr,
)
from beyond_backprop.algorithms.mf_math import local_cross_entropy, projection_logits


def test_ff_hinton_inputs_preserve_label_embedding_and_detach_outputs():
    images = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    labels = torch.tensor([0, 1])
    torch.manual_seed(4)
    positive, negative, neutral = generate_hinton_inputs(
        images, labels, num_classes=2, device=torch.device("cpu")
    )

    assert positive[:, :2].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert torch.allclose(neutral[:, :2], torch.full((2, 2), 0.1))
    assert all(not torch.equal(negative[i, :2], positive[i, :2]) for i in range(2))
    assert not positive.requires_grad
    assert not negative.requires_grad
    assert not neutral.requires_grad


def test_ff_schedule_and_goodness_aggregation_match_protocol():
    assert linear_cooldown_lr(1.0, epoch=0, total_epochs=4) == 1.0
    assert linear_cooldown_lr(1.0, epoch=2, total_epochs=4) == 1.0
    assert linear_cooldown_lr(1.0, epoch=3, total_epochs=4) == 0.5
    goodness = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    assert torch.equal(
        aggregate_goodness(goodness, start_layer=0), goodness[0] + goodness[1]
    )
    assert torch.equal(aggregate_goodness([goodness[0]]), goodness[0])


def test_cafo_aggregation_supports_sum_average_last_and_rejects_unknown():
    outputs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
    assert torch.equal(
        aggregate_predictor_outputs(outputs, "sum"), torch.tensor([[4.0, 6.0]])
    )
    assert torch.equal(
        aggregate_predictor_outputs(outputs, "average"), torch.tensor([[2.0, 3.0]])
    )
    assert torch.equal(
        aggregate_predictor_outputs(outputs, "last"), torch.tensor([[3.0, 4.0]])
    )
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate_predictor_outputs(outputs, "median")


def test_mf_projection_and_local_loss_are_differentiable():
    activation = torch.tensor([[1.0, 2.0]], requires_grad=True)
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    logits = projection_logits(activation, projection)
    loss = local_cross_entropy(activation, projection, torch.tensor([1]))
    assert torch.equal(logits, torch.tensor([[1.0, 2.0]]))
    loss.backward()
    assert activation.grad is not None
    assert projection.grad is not None
