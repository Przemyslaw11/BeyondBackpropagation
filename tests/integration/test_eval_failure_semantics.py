"""Evaluation failure semantics (EVAL-001 contract).

Any failed batch during evaluation aborts evaluation with an exception instead
of substituting predictions or NaN metrics. The exception propagates through
``ExperimentRunner.run``, which records ``RunStatus.FAILED`` (and the CLI maps
that to a non-zero exit code). The superseded legacy behavior was pinned by
characterization tests in commit 52f9d61 before this contract changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms.base import AlgorithmAdapter, evaluation_result
from beyond_backprop.algorithms.cafo import evaluate_cafo_model
from beyond_backprop.algorithms.ff import evaluate_ff_model
from beyond_backprop.algorithms.mf import evaluate_mf_model
from beyond_backprop.architectures.ff_mlp import FF_MLP
from beyond_backprop.config.loader import experiment_config_from_mapping
from beyond_backprop.contracts import ResourceSnapshot, RunStatus, TrainingResult
from beyond_backprop.training.runner import ExperimentRunner

CPU = torch.device("cpu")


def _loader(image_dim: int = 16, samples: int = 4) -> DataLoader:
    dataset = TensorDataset(
        torch.rand(samples, image_dim),
        torch.ones(samples, dtype=torch.long),  # nonzero labels: substitution shows up
    )
    return DataLoader(dataset, batch_size=2)


# ---------------------------------------------------------------------------
# EVAL-001: failed batches abort evaluation
# ---------------------------------------------------------------------------


class _BrokenForwardFF(FF_MLP):
    """FF_MLP whose goodness forward always fails."""

    def forward_goodness_per_layer(self, x_input: torch.Tensor):  # noqa: D102
        raise RuntimeError("injected forward failure")


def test_ff_forward_failure_aborts_evaluation():
    model = _BrokenForwardFF({}, CPU, input_dim=16, hidden_dims=[8], num_classes=2)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        evaluate_ff_model(model, _loader(), CPU)


class _ExplodingPredictor(nn.Module):
    def forward(self, block_output: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected predictor failure")


class _CafoStub(nn.Module):
    def __init__(self, num_blocks: int = 1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(num_blocks)])

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [torch.zeros(images.shape[0], 3) for _ in range(len(self.blocks))]


def test_cafo_predictor_failure_aborts_evaluation():
    with pytest.raises(RuntimeError, match="injected predictor failure"):
        evaluate_cafo_model(
            model=_CafoStub(),
            data_loader=_loader(image_dim=3),
            device=CPU,
            criterion=None,
            predictors=nn.ModuleList([_ExplodingPredictor()]),
            aggregation_method="sum",
        )


def test_cafo_last_aggregation_guard_raises_instead_of_nan():
    # 'last' aggregation requires predictors == blocks; two blocks vs one
    # predictor violates it.
    with pytest.raises(ValueError, match="'last' aggregation"):
        evaluate_cafo_model(
            model=_CafoStub(num_blocks=2),
            data_loader=_loader(image_dim=3),
            device=CPU,
            criterion=None,
            predictors=nn.ModuleList([nn.Identity()]),
            aggregation_method="last",
        )


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
    ("num_matrices", "activation_len", "match"),
    [
        (0, 2, "out of bounds"),
        (2, 1, "too short"),
    ],
    ids=["projection-index-guard", "short-activation-guard"],
)
def test_mf_domain_guards_raise_instead_of_degrading_results(
    num_matrices: int, activation_len: int, match: str
):
    model = _MfStub(num_matrices, activation_len)
    with pytest.raises(ValueError, match=match):
        evaluate_mf_model(
            model=model,
            data_loader=_loader(image_dim=3),
            device=CPU,
            input_adapter=lambda tensor: tensor,
        )


# ---------------------------------------------------------------------------
# Runner propagation: evaluation failure -> RunStatus.FAILED
# ---------------------------------------------------------------------------


class _EvalDelegatingAdapter(AlgorithmAdapter):
    """Succeeds at fit, delegates evaluation to the real FF eval routine."""

    name = "ff"

    def fit(self, context):
        return TrainingResult(status=RunStatus.SUCCEEDED)

    def evaluate(self, model, loader, context):
        return evaluation_result(
            evaluate_ff_model(model, loader, torch.device(context.device))
        )


class _Registry:
    def __init__(self, adapter: AlgorithmAdapter) -> None:
        self.adapter = adapter

    def build(self, name: str) -> AlgorithmAdapter:
        return self.adapter


@dataclass
class _Tracker:
    finished: RunStatus | None = None

    def log_config(self, config):
        pass

    def log_metrics(self, metrics):
        pass

    def log_artifact(self, path):
        pass

    def finish(self, status):
        self.finished = status


class _Monitor:
    def start(self):
        pass

    def stop(self):
        return ResourceSnapshot(source="synthetic")


def _runner_config():
    return experiment_config_from_mapping(
        {
            "experiment_name": "eval-failure",
            "general": {"seed": 3, "device": "cpu", "backend": "local"},
            "algorithm": {"name": "FF"},
            "model": {"name": "FF_MLP", "params": {"hidden_dims": [3]}},
            "data": {
                "name": "MNIST",
                "root": "/tmp/offline",
                "download": False,
                "val_split": 0.2,
                "num_classes": 2,
                "input_channels": 1,
                "image_size": 2,
            },
            "data_loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "optimizer": {"type": "AdamW", "lr": 0.01, "weight_decay": 0.0},
            "training": {"epochs": 1, "early_stopping_enabled": False},
            "algorithm_params": {},
            "checkpointing": {},
            "monitoring": {},
            "tracking": {},
        }
    )


def test_runner_marks_run_failed_when_evaluation_raises(tmp_path):
    dataset = TensorDataset(torch.zeros(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    model = _BrokenForwardFF({}, CPU, input_dim=4, hidden_dims=[4], num_classes=2)
    tracker = _Tracker()
    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: model,
        algorithm_registry=_Registry(_EvalDelegatingAdapter()),
        tracker=tracker,
        resource_monitor=_Monitor(),
        artifact_dir=tmp_path,
    ).run(_runner_config())
    assert result.status is RunStatus.FAILED
    assert "injected forward failure" in (result.error or "")
    assert tracker.finished is RunStatus.FAILED
