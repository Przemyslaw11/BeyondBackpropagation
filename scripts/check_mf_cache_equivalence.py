"""Compares the three MF activation-cache strategies on a short two-layer run.

Caching only pays off if it is a pure speed/memory trade: the loss trajectory and the
final weights must match recomputation exactly at a fixed seed. Run this after touching
anything in the MF activation-cache path.
"""

import argparse
import logging
import os
import resource
import sys
import time
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import mf  # noqa: E402
from src.architectures.mf_mlp import MF_MLP  # noqa: E402
from src.data_utils.datasets import get_dataloaders  # noqa: E402
from src.utils.helpers import set_seed  # noqa: E402

STRATEGIES = ("recompute", "cache_device", "cache_host")


def _peak_rss_mib() -> float:
    """Returns the process high-water RSS; ru_maxrss is bytes on macOS, KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**2) if sys.platform == "darwin" else raw / 1024


def run(strategy: str, args: argparse.Namespace) -> Dict[str, object]:
    """Trains a two-layer MF_MLP under one cache strategy and reports its trajectory."""
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, _ = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_root=args.data_root,
        seed=args.seed,
        num_workers=0,
        pin_memory=False,
    )
    train_loader = _truncate(train_loader, args.train_batches)

    model = MF_MLP(
        input_dim=args.input_dim,
        hidden_dims=[args.width] * args.depth,
        num_classes=10,
    ).to(device)

    config = {
        "algorithm": {"name": "MF"},
        "algorithm_params": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer_type": "Adam",
            "activation_cache": strategy,
        },
        "early_stopping": {"enabled": False, "max_epochs": args.epochs},
    }

    losses: List[float] = []
    original_loss_fn = mf.mf_local_loss_fn

    def recording_loss_fn(*fn_args, **fn_kwargs):
        loss = original_loss_fn(*fn_args, **fn_kwargs)
        losses.append(float(loss.detach()))
        return loss

    mf.mf_local_loss_fn = recording_loss_fn
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        mf.train_mf_model(
            model=model,
            train_loader=train_loader,
            config=config,
            device=device,
            input_adapter=lambda x: x.view(x.size(0), -1),
            val_loader=val_loader,
        )
    finally:
        mf.mf_local_loss_fn = original_loss_fn
    wall_time = time.perf_counter() - started

    return {
        "strategy": strategy,
        "wall_time_sec": wall_time,
        "losses": losses,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "peak_torch_alloc_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else float("nan")
        ),
        "peak_rss_mib": _peak_rss_mib(),
    }


def _truncate(loader: torch.utils.data.DataLoader, num_batches: int):
    """Shrinks the training set so the comparison stays short."""
    subset = torch.utils.data.Subset(
        loader.dataset, list(range(num_batches * loader.batch_size))
    )
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=loader.drop_last,
    )


def main() -> int:
    """Runs all three strategies and reports trajectory and weight differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=784)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    results = {strategy: run(strategy, args) for strategy in STRATEGIES}
    reference = results["recompute"]

    print(f"\n{'strategy':<14}{'wall_s':>10}{'speedup':>10}{'peak_alloc_MiB':>17}{'peak_RSS_MiB':>15}")
    for strategy in STRATEGIES:
        r = results[strategy]
        speedup = reference["wall_time_sec"] / r["wall_time_sec"]
        print(
            f"{strategy:<14}{r['wall_time_sec']:>10.2f}{speedup:>10.2f}x"
            f"{r['peak_torch_alloc_mib']:>16.1f}{r['peak_rss_mib']:>15.1f}"
        )

    ok = True
    print()
    for strategy in STRATEGIES[1:]:
        r = results[strategy]
        if len(r["losses"]) != len(reference["losses"]):
            print(f"FAIL {strategy}: {len(r['losses'])} loss steps vs {len(reference['losses'])}")
            ok = False
            continue
        max_loss_diff = max(
            abs(a - b) for a, b in zip(r["losses"], reference["losses"])
        )
        max_weight_diff = max(
            float((r["state"][k] - reference["state"][k]).abs().max())
            for k in reference["state"]
        )
        verdict = "OK" if max_loss_diff == 0.0 and max_weight_diff == 0.0 else "DIFFERS"
        ok = ok and verdict == "OK"
        print(
            f"{verdict} {strategy} vs recompute: {len(r['losses'])} loss steps, "
            f"max|dloss|={max_loss_diff:.3e}, max|dweight|={max_weight_diff:.3e}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
