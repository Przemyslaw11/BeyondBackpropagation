#!/usr/bin/env python3
"""W3/ERRATA: audit stage 0's dead forward and bound its energy share.

Two independent checks:

1. Run the production ``train_mf_matrix_only`` on a tiny MF_MLP with forward
   hooks on every W layer.  One training epoch and one validation epoch must
   call W1..WL even though M0 reads only a_0.
2. Compute the canonical 3x2000/CIFAR FLOP split between the dead full forward
   (train + validation) and the necessary M0 head update.  The M0 stage energy
   from W0b is multiplied by the dead-forward FLOP share to get a lower bound on
   pure implementation waste; this is not a measured sub-stage joule split.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.mf import train_mf_matrix_only
from src.architectures.mf_mlp import MF_MLP

INPUT_DIM = 3072
HIDDEN = [2000, 2000, 2000]
TRAIN_SAMPLES = 45_000
VAL_SAMPLES = 5_000
M0_ENERGY_WH = {"cifar10": 3.986442351914517, "cifar100": 3.567443175040735}


def production_path_check() -> dict:
    torch.manual_seed(0)
    model = MF_MLP(6, [4, 4], 3, activation="relu")
    calls = {f"W{i}": 0 for i in range(1, 3)}
    handles = []
    for i in range(2):
        linear = model.layers[2 * i]
        handles.append(
            linear.register_forward_hook(
                lambda m, inp, out, name=f"W{i + 1}": calls.__setitem__(
                    name, calls[name] + 1
                )
            )
        )

    train_x = torch.randn(8, 6)
    train_y = torch.randint(0, 3, (8,))
    val_x = torch.randn(4, 6)
    val_y = torch.randint(0, 3, (4,))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=4)

    optimizer = torch.optim.Adam([model.get_projection_matrix(0)], lr=1e-3)
    train_mf_matrix_only(
        model=model,
        matrix_index=0,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        epochs=1,
        device=torch.device("cpu"),
        input_adapter=lambda x: x,
        early_stopping_config={"enabled": True, "patience": 5, "min_delta": 0.0},
        val_loader=val_loader,
        log_interval=9999,
        step_ref=[-1],
        gpu_handle=None,
        nvml_active=False,
    )
    for h in handles:
        h.remove()
    return calls


def canonical_flop_split(num_classes: int) -> dict:
    full_fwd_per_sample = 2 * (
        INPUT_DIM * HIDDEN[0]
        + HIDDEN[0] * HIDDEN[1]
        + HIDDEN[1] * HIDDEN[2]
    )
    dead_train = full_fwd_per_sample * TRAIN_SAMPLES
    dead_val = full_fwd_per_sample * VAL_SAMPLES
    # Linear head: forward 2*N*D*C, weight-gradient 2*N*D*C, bias N*C.
    head_train_fwd = 2 * TRAIN_SAMPLES * INPUT_DIM * num_classes
    head_train_bwd = 2 * TRAIN_SAMPLES * INPUT_DIM * num_classes
    head_train_bias = TRAIN_SAMPLES * num_classes
    head_val = 2 * VAL_SAMPLES * INPUT_DIM * num_classes
    necessary = head_train_fwd + head_train_bwd + head_train_bias + head_val
    dead = dead_train + dead_val
    return dict(
        dead_train_flops=dead_train,
        dead_val_flops=dead_val,
        dead_flops=dead,
        necessary_flops=necessary,
        dead_share=dead / (dead + necessary),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root.parent / "jpdc_data" / "g1" / "w3_cut"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = production_path_check()
    splits = {
        dataset: canonical_flop_split(10 if dataset == "cifar10" else 100)
        for dataset in ("cifar10", "cifar100")
    }
    rows = []
    for dataset, split in splits.items():
        e0 = M0_ENERGY_WH[dataset]
        dead_wh = e0 * split["dead_share"]
        rows.append(
            dict(
                dataset=dataset,
                m0_energy_wh=e0,
                dead_share=split["dead_share"],
                dead_energy_lower_bound_wh=dead_wh,
                head_energy_upper_bound_wh=e0 - dead_wh,
                dead_train_tflop=split["dead_train_flops"] / 1e12,
                dead_val_tflop=split["dead_val_flops"] / 1e12,
                necessary_gflop=split["necessary_flops"] / 1e9,
            )
        )

    payload = dict(
        production_path_hook_calls=calls,
        expected_calls="W1 and W2 each called in 2 train batches + 1 val batch",
        rows=rows,
    )
    (out_dir / "w3_stage0_audit.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Stage-0 forward-waste audit", ""]
    lines.append("## Production path check")
    lines.append("")
    lines.append(
        "`train_mf_matrix_only` was run for one epoch on a tiny MF_MLP with "
        "forward hooks on every W layer. Hook calls: "
        + ", ".join(f"`{k}={v}`" for k, v in calls.items())
        + ". The function runs a full forward through W1..WL for both the two "
        "training batches and the validation batch, then reads only `a_0` for the "
        "M0 head. This is the production code path, not a reimplementation."
    )
    lines.append("")
    lines.append("## Canonical FLOP split (3×2000 CIFAR MLP)")
    lines.append("")
    lines.append(
        "| dataset | M0 energy Wh | dead-forward FLOP share | dead energy lower bound Wh | head+protocol upper bound Wh |"
    )
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['m0_energy_wh']:.2f} | "
            f"{100*r['dead_share']:.1f}% | {r['dead_energy_lower_bound_wh']:.2f} | "
            f"{r['head_energy_upper_bound_wh']:.2f} |"
        )
    lines.append("")
    lines.append(
        "The dead forward is 99.6% (C10) / 96.0% (C100) of M0-stage FLOPs. "
        "Multiplying the measured M0 stage energy by that FLOP share gives a "
        "lower bound of **3.97 Wh (C10) / 3.43 Wh (C100)** of pure implementation "
        "waste. The complement (0.02 / 0.14 Wh) is an upper bound on the necessary "
        "M0 head plus protocol overhead; the exact joule split is not observable "
        "without sub-stage NVML. Use the ex-M0 multipliers in `w3_cut_ex_m0.csv` "
        "for the conservative, implementation-independent correction."
    )
    (out_dir / "w3_stage0_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
