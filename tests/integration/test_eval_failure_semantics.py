"""Evaluation failure semantics.

The ``*_legacy_*`` tests characterize the OLD contract: internal eval failures
were silently substituted with class-0 predictions or NaN metrics. WP1
supersedes them (decision EVAL-001 in docs/refactoring-decisions.md); they pin
the superseded behavior so the change is deliberate and reviewable.

The remaining tests assert the NEW contract: any failed batch aborts evaluation
with an exception, which propagates through ``ExperimentRunner.run`` into
``RunStatus.FAILED``.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms.cafo import evaluate_cafo_model
from beyond_backprop.algorithms.ff import evaluate_ff_model
from beyond_backprop.algorithms.mf import evaluate_mf_model
from beyond_backprop.architectures.ff_mlp import FF_MLP

CPU = torch.device("cpu")


def _loader(image_dim: int = 16, samples: int = 4) -> DataLoader:
    dataset = TensorDataset(
        torch.rand(samples, image_dim),
        torch.ones(samples, dtype=torch.long),  # nonzero labels: substitution shows up
    )
    return DataLoader(dataset, batch_size=2)


# ---------------------------------------------------------------------------
# Legacy contract (superseded by EVAL-001)
# ---------------------------------------------------------------------------


class _BrokenForwardFF(FF_MLP):
    """FF_MLP whose goodness forward always fails."""

    def forward_goodness_per_layer(self, x_input: torch.Tensor):  # noqa: D102
        raise RuntimeError("injected forward failure")


def test_legacy_ff_forward_failure_substitutes_class_zero_predictions():
    model = _BrokenForwardFF({}, CPU, input_dim=16, hidden_dims=[8], num_classes=2)
    result = evaluate_ff_model(model, _loader(), CPU)
    assert math.isnan(result["eval_loss"])
    # All samples predicted as class 0 although every label is class 1.
    assert result["eval_accuracy"] == 0.0


class _ExplodingPredictor(nn.Module):
    def forward(self, block_output: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected predictor failure")


class _CafoStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3)])

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [torch.zeros(images.shape[0], 3)]


def test_legacy_cafo_predictor_failure_returns_nan_metrics():
    result = evaluate_cafo_model(
        model=_CafoStub(),
        data_loader=_loader(image_dim=3),
        device=CPU,
        criterion=None,
        predictors=nn.ModuleList([_ExplodingPredictor()]),
        aggregation_method="sum",
    )
    assert math.isnan(result["eval_accuracy"])
    assert math.isnan(result["eval_loss"])


class _MfStub(nn.Module):
    def __init__(
        self,
        num_projection_matrices: int,
        activation_list_length: int,
    ) -> None:
        super().__init__()
        self.num_hidden_layers = 1
        self.projection_matrices = [
            nn.Linear(3, 2).weight for _ in range(num_projection_matrices)
        ]
        self._activation_list_length = activation_list_length

    def get_projection_matrix(self, index: int) -> torch.Tensor:
        return self.projection_matrices[index]

    def forward_with_intermediate_activations(
        self, eval_input: torch.Tensor
    ) -> list[torch.Tensor]:
        return [eval_input] * self._activation_list_length


@pytest.mark.parametrize(
    ("num_matrices", "activation_len", "expected_accuracy"),
    [
        (0, 2, float("nan")),  # projection index guard -> NaN dict
        (2, 1, 0.0),  # short activation list -> batch skipped via continue
    ],
    ids=["projection-index-guard", "short-activation-skip"],
)
def test_legacy_mf_index_errors_yield_degenerate_results(
    num_matrices: int, activation_len: int, expected_accuracy: float
):
    model = _MfStub(num_matrices, activation_len)
    result = evaluate_mf_model(
        model=model,
        data_loader=_loader(image_dim=3),
        device=CPU,
        input_adapter=lambda tensor: tensor,
    )
    assert result["eval_accuracy"] == pytest.approx(
        expected_accuracy, nan_ok=True, abs=0.0
    )
