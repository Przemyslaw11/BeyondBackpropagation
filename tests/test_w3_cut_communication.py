"""Guards the W3 cut-communication unit and hook counts."""

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jpdc" / "w3_cut_communication.py"
spec = importlib.util.spec_from_file_location("w3_cut", SCRIPT)
w3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(w3)


def test_cut_unit_is_one_epoch_tensor():
    # 45,000 x 2,000 x 4 B = 0.36 GB; the batch scale cancels.
    assert w3.BATCH * w3.HIDDEN[0] * 4 * w3.TRAIN_SAMPLES / w3.BATCH / w3.GB == 0.36


def test_hooks_count_one_tensor_per_cut_and_direction():
    model = w3.build(num_classes=10)
    counter, handles = w3.attach(model, backward=True)
    try:
        x = torch.randn(2, w3.INPUT_DIM)
        loss = model(x).sum()
        loss.backward()
    finally:
        w3.detach(handles)

    expected = 2 * w3.HIDDEN[0] * 4
    assert counter.forward[1] == expected
    assert counter.backward[1] == expected
    assert counter.forward[3] == expected
    assert counter.backward[3] == expected
