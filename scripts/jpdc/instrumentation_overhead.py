#!/usr/bin/env python3
"""Measure the post-training diagnostic-probe overhead, per W1 arm.

This is an engineering check before the freeze, not a W1 result.  It uses a
fixed synthetic 5,000-row CIFAR-shaped probe set and the exact
``compute_secondary_endpoints`` implementation that runs in every W1 job.  It
records (a) the probe seconds returned in the run JSON field
``secondary_eval_seconds``, and (b) a SHA-256 fingerprint of the model state
before and after the probes.  The state must be identical.

The real W1 per-run seconds are the ``secondary_eval_seconds`` values in each
run JSON; this script only establishes the expected order of magnitude and the
inertness in a controlled CPU environment.  It does not touch results/runs.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.architectures.mf_mlp import MF_MLP
from src.training.secondary_endpoints import compute_secondary_endpoints

SEED = 20261022
INPUT_DIM = 32 * 32 * 3
HIDDEN_DIMS = [2000, 2000, 2000]
NUM_CLASSES = 10
N_ROWS = 5_000
BATCH_SIZE = 512

ARMS = {
    "bp": "bp",
    "mf_joint": "mf_joint",
    "mf_simul": "mf_simul",
    "mf_seq": "mf",
    "mf_global": "mf_global",
    "mf_nom0": "mf",
}


def _state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model(arm: str) -> nn.Module:
    torch.manual_seed(SEED)
    if arm == "bp":
        return nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[2], NUM_CLASSES),
        )
    return MF_MLP(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        activation="ReLU",
    )


def _loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(N_ROWS, INPUT_DIM, generator=generator)
    targets = torch.randint(0, NUM_CLASSES, (N_ROWS,), generator=generator)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    # The probe code caps at 5,000 rows, so reusing the same loader for all
    # splits keeps this a pure overhead measurement rather than a dataset test.
    return loader, loader, loader


def _config(arm: str) -> Dict[str, object]:
    return {
        "experiment_name": f"instrumentation_{arm}",
        "algorithm": {"name": ARMS[arm]},
        "data": {"name": "CIFAR10", "num_classes": NUM_CLASSES},
        "monitoring": {"enabled": False},
        "profiling": {"enabled": False},
        "logging": {"wandb": {"use_wandb": False}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("../jpdc_data/g1/instrumentation"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loader, val_loader, test_loader = _loaders()
    rows: List[Dict[str, object]] = []
    for arm, algo_name in ARMS.items():
        model = _model(arm)
        cfg = _config(arm)
        state_before = _state_fingerprint(model)
        times: List[float] = []
        returned_seconds: List[float] = []
        scopes: List[str] = []
        state_unchanged = True
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = compute_secondary_endpoints(
                model=model,
                train_loader=loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=cfg,
                device=torch.device("cpu"),
                algo_name=algo_name,
                input_adapter=lambda x: x,
                test_loss=1.0,
                test_accuracy=50.0,
            )
            times.append(time.perf_counter() - started)
            returned_seconds.append(float(result.get("secondary_eval_seconds", float("nan"))))
            scopes.append(str(result.get("secondary_eval_scope", "")))
            state_unchanged = state_unchanged and (_state_fingerprint(model) == state_before)
        rows.append(
            {
                "arm": arm,
                "algo_name": algo_name,
                "repeats": args.repeats,
                "mean_wall_seconds": float(sum(times) / len(times)),
                "min_wall_seconds": float(min(times)),
                "max_wall_seconds": float(max(times)),
                "mean_json_seconds": float(sum(returned_seconds) / len(returned_seconds)),
                "state_unchanged": state_unchanged,
                "scope": scopes[-1] if scopes else "",
            }
        )
        print(
            f"{arm}: mean {rows[-1]['mean_wall_seconds']:.3f}s "
            f"(json {rows[-1]['mean_json_seconds']:.3f}s), "
            f"state_unchanged={state_unchanged}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "instrumentation_overhead.csv", index=False)
    lines = [
        "# Instrumentation overhead — synthetic CPU check",
        "",
        "Not a W1 result. A fixed 5,000-row CIFAR-shaped probe set, canonical "
        "3x2000 MLP, exact post-training endpoint implementation; "
        f"{args.repeats} repeat(s) per arm. The authoritative per-run value in "
        "W1 is the `secondary_eval_seconds` field in each run JSON.",
        "",
        "| arm | mean wall s | min s | max s | mean JSON s | state unchanged | scope |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| {row['arm']} | {row['mean_wall_seconds']:.3f} | "
            f"{row['min_wall_seconds']:.3f} | {row['max_wall_seconds']:.3f} | "
            f"{row['mean_json_seconds']:.3f} | {row['state_unchanged']} | "
            f"{row['scope']} |"
        )
    lines.append("")
    (args.out_dir / "instrumentation_overhead.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir / 'instrumentation_overhead.csv'}")


if __name__ == "__main__":
    main()
