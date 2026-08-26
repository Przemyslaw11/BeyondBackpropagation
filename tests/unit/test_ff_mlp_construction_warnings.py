"""FF_MLP construction must not emit the autograd-instantiation warning (MIG-005)."""

from __future__ import annotations

import warnings

import torch

from beyond_backprop.architectures.ff_mlp import FF_MLP, ReLU_full_grad


def test_ff_mlp_construction_raises_no_deprecation_warning():
    config = {
        "model": {"params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2},
        "algorithm_params": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model = FF_MLP(config, device=None, input_dim=4, hidden_dims=[3], num_classes=2)
    assert model.act_fn_train is ReLU_full_grad
    probe = torch.tensor([-1.0, 0.0, 2.0])
    assert model.act_fn_train.apply(probe).tolist() == [0.0, 0.0, 2.0]
