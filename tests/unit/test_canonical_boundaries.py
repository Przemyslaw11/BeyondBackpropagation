from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from beyond_backprop.config import ConfigValidationError, validate_mapping
from beyond_backprop.data.loaders import seed_worker
from beyond_backprop.runtime import collect_run_metadata
from beyond_backprop.utils import canonical_imports


def test_strict_validation_rejects_coercible_scalar_values() -> None:
    with pytest.raises(ConfigValidationError, match="batch_size must be an integer"):
        validate_mapping({"data_loader": {"batch_size": "8"}})
    with pytest.raises(ConfigValidationError, match="download must be a boolean"):
        validate_mapping({"data": {"download": "false"}})
    with pytest.raises(
        ConfigValidationError, match="algorithm.name must be of type str"
    ):
        validate_mapping({"algorithm": {"name": False}})


def test_worker_seed_is_reproducible_across_python_and_numpy() -> None:
    torch.manual_seed(11)
    seed_worker(0)
    first = (random.random(), float(np.random.random()))
    torch.manual_seed(11)
    seed_worker(0)
    second = (random.random(), float(np.random.random()))
    assert first == second


def test_metadata_contains_runtime_identity_and_config_hash() -> None:
    metadata = collect_run_metadata("test", device="cpu", seed=9, config_hash="abc")
    payload = metadata.to_dict()
    assert payload["python_version"]
    assert payload["torch_version"]
    assert "torchvision_version" in payload
    assert payload["config_hash"] == "abc"
    assert "cuda_available" in payload
    assert "mps_available" in payload
    assert payload["hostname"]


def test_canonical_import_tree_is_explicit() -> None:
    modules = canonical_imports()
    assert "beyond_backprop.config" in modules
    assert "beyond_backprop.experiment" in modules
    assert "beyond_backprop.evaluation" in modules
