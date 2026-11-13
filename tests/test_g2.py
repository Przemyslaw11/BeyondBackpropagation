"""G2 harness checks: GPU augmentation semantics, arm isolation, BP MF-init control, CPU smoke."""

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.mf_mlp import MF_MLP
from src.training.engine import _apply_mf_mlp_init

_spec = importlib.util.spec_from_file_location(
    "g2_sweep", Path(__file__).resolve().parents[1] / "scripts/jpdc/g2_sweep.py"
)
g2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2)


def test_gpu_augment_is_reflect_crop_or_its_flip():
    x = torch.randn(16, 3, 32, 32)
    out = g2.gpu_augment(x, torch.Generator().manual_seed(0))
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    flips = 0
    for i in range(16):
        crops = [padded[i, :, a : a + 32, b : b + 32] for a in range(9) for b in range(9)]
        plain = any(torch.equal(out[i], c) for c in crops)
        flipped = any(torch.equal(out[i], c.flip(-1)) for c in crops)
        assert plain or flipped
        flips += flipped and not plain
    assert 0 < flips < 16  # both branches exercised at this seed


def test_mf_stage_step_updates_only_its_stage():
    torch.manual_seed(0)
    step = g2.make_arm("mf_stage2", "cifar10", 32, 3, torch.device("cpu"))
    model = step.model
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    step(torch.randn(8, 3072), torch.randint(0, 10, (8,)))
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {"layers.2.weight", "layers.2.bias", "projection_matrices.2"}


def test_bp_mf_init_matches_mf_mlp_hidden_init():
    torch.manual_seed(0)
    layers = [nn.Linear(3072, 2000), nn.ReLU(), nn.Linear(2000, 2000), nn.ReLU(), nn.Linear(2000, 10)]
    _apply_mf_mlp_init(layers, "relu")
    mf = MF_MLP(input_dim=3072, hidden_dims=[2000, 2000], num_classes=10)
    for bp_lin, mf_lin in [(layers[0], mf.layers[0]), (layers[2], mf.layers[2])]:
        assert math.isclose(bp_lin.weight.std().item(), mf_lin.weight.std().item(), rel_tol=0.01)
        assert bp_lin.bias.abs().max() == 0
    assert layers[4].bias.abs().max() == 0
    default = nn.Linear(3072, 2000)
    assert layers[0].weight.std() > 2.4 * default.weight.std()  # the documented 2.45x gap


def test_sweep_smoke_cpu(tmp_path):
    out = tmp_path / "r0.json"
    rec = g2.main(["--dataset", "cifar10", "--repeat", "0", "--out", str(out), "--synthetic",
                   "--batches", "8", "16", "--widths", "16", "--idle-sec", "0",
                   "--warmup-sec", "0.01", "--measure-sec", "0.01"])
    assert out.exists()
    assert len(rec["units"]) == 2 * 1 * 5
    assert {u["arm"] for u in rec["units"]} == {"bp", "mf_simul", "mf_stage1", "mf_stage2", "mf_stage3"}
    assert all(u["steps"] >= 5 and u["counter_energy_j"] is None for u in rec["units"])
    orders = [(u["batch"], u["arm"]) for u in rec["units"]]
    rec1 = g2.main(["--dataset", "cifar10", "--repeat", "1", "--out", str(tmp_path / "r1.json"),
                    "--synthetic", "--batches", "8", "16", "--widths", "16", "--idle-sec", "0",
                    "--warmup-sec", "0.01", "--measure-sec", "0.01"])
    assert orders != [(u["batch"], u["arm"]) for u in rec1["units"]]  # order differs by repeat


def test_engine_bp_builder_honours_bp_init_flag():
    from src.training.engine import _create_mlp_model

    params = {"input_dim": 64, "hidden_dims": [32, 32], "activation": "ReLU", "bias": True}
    plain = _create_mlp_model("mf_mlp", True, {}, torch.device("cpu"), dict(params), 10)
    mfinit = _create_mlp_model("mf_mlp", True, {}, torch.device("cpu"), {**params, "bp_init": "mf"}, 10)
    assert plain[0].bias.abs().max() > 0
    assert all(m.bias.abs().max() == 0 for m in mfinit if isinstance(m, nn.Linear))
    mf_arm = _create_mlp_model("mf_mlp", False, {}, torch.device("cpu"), {**params, "num_classes": 10, "bp_init": "mf"}, 10)
    assert isinstance(mf_arm, MF_MLP)  # the flag is ignored outside the BP path
