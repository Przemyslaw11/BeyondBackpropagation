#!/usr/bin/env python3
"""W6 exploratory toy numerics: local/staged credit assignment on a deep linear task.

This is **exploratory**, not a confirmatory W6 protocol. No W6 preregistration
exists in the repository yet; this script only asks whether the qualitative
staging story is numerically plausible on a tiny CPU problem. It must not be
cited as evidence for a W1/W2/W3/W4 hypothesis.

Task: x ~ N(0, I_d), y = T x with a rank-r teacher T. The student is a
three-layer linear bottleneck (d -> h -> h -> d). We compare:

  BP          end-to-end backpropagation.
  MF-Simul    per-layer local heads, detached cut inputs, joint optimisation.
  MF-Seq      greedy stagewise training with a frozen prefix.
  Seq-Global  stagewise training with an unfrozen prefix.

The reported quantity is held-out MSE. We also report the cosine similarity
between the global BP gradient and the first-stage local-head gradient at
initialisation, as a diagnostic of local/global signal alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def make_task(seed: int, n_train: int = 512, n_test: int = 512, d: int = 8, rank: int = 2):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(d, rank, generator=gen) / np.sqrt(d)
    b = torch.randn(rank, d, generator=gen) / np.sqrt(rank)
    teacher = a @ b
    x_train = torch.randn(n_train, d, generator=gen)
    x_test = torch.randn(n_test, d, generator=gen)
    return x_train, x_train @ teacher.T, x_test, x_test @ teacher.T


def init_student(seed: int, d: int, h: int):
    # Reset the global RNG so every method's local heads start identically.
    torch.manual_seed(20_000 + seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    def lin(i, o):
        layer = nn.Linear(i, o, bias=False)
        with torch.no_grad():
            layer.weight.normal_(0.0, 0.1, generator=gen)
        return layer
    return lin(d, h), lin(h, h), lin(h, d)


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def train_params(params, loss_fn, steps: int, lr: float = 0.01):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()


def eval_mse(w1, w2, w3, x, y):
    with torch.no_grad():
        return float(mse(w3(w2(w1(x))), y))


def run_bp(seed, x, y, xt, yt, d, h, steps):
    w1, w2, w3 = init_student(seed, d, h)
    train_params(list(w1.parameters()) + list(w2.parameters()) + list(w3.parameters()),
                 lambda: mse(w3(w2(w1(x))), y), steps)
    return eval_mse(w1, w2, w3, xt, yt)


def run_mf_seq(seed, x, y, xt, yt, d, h, steps):
    w1, w2, w3 = init_student(seed, d, h)
    h1 = nn.Linear(h, d, bias=False)
    h2 = nn.Linear(h, d, bias=False)
    per = max(1, steps // 3)
    train_params(list(w1.parameters()) + list(h1.parameters()),
                 lambda: mse(h1(w1(x)), y), per)
    with torch.no_grad():
        x1 = w1(x)
    train_params(list(w2.parameters()) + list(h2.parameters()),
                 lambda: mse(h2(w2(x1)), y), per)
    with torch.no_grad():
        x2 = w2(x1)
    train_params(list(w3.parameters()), lambda: mse(w3(x2), y), per)
    return eval_mse(w1, w2, w3, xt, yt)


def run_mf_simul(seed, x, y, xt, yt, d, h, steps):
    w1, w2, w3 = init_student(seed, d, h)
    h1 = nn.Linear(h, d, bias=False)
    h2 = nn.Linear(h, d, bias=False)
    params = (list(w1.parameters()) + list(w2.parameters()) + list(w3.parameters())
              + list(h1.parameters()) + list(h2.parameters()))

    def loss():
        z1 = w1(x)
        z2 = w2(z1.detach())
        z3 = w3(z2.detach())
        return (mse(h1(z1), y) + mse(h2(z2), y) + mse(z3, y)) / 3.0

    train_params(params, loss, steps)
    return eval_mse(w1, w2, w3, xt, yt)


def run_seq_global(seed, x, y, xt, yt, d, h, steps):
    w1, w2, w3 = init_student(seed, d, h)
    h1 = nn.Linear(h, d, bias=False)
    h2 = nn.Linear(h, d, bias=False)
    per = max(1, steps // 3)
    train_params(list(w1.parameters()) + list(h1.parameters()),
                 lambda: mse(h1(w1(x)), y), per)
    train_params(list(w1.parameters()) + list(w2.parameters()) + list(h2.parameters()),
                 lambda: mse(h2(w2(w1(x))), y), per)
    train_params(list(w1.parameters()) + list(w2.parameters()) + list(w3.parameters()),
                 lambda: mse(w3(w2(w1(x))), y), per)
    return eval_mse(w1, w2, w3, xt, yt)


def gradient_cosine(seed, x, y, d, h):
    """Cosine between global BP and first-stage local-head gradients at init."""
    w1, w2, w3 = init_student(seed, d, h)
    h1 = nn.Linear(h, d, bias=False)

    w1.zero_grad()
    mse(w3(w2(w1(x))), y).backward()
    g_global = w1.weight.grad.detach().flatten().clone()

    w1.zero_grad()
    mse(h1(w1(x)), y).backward()
    g_local = w1.weight.grad.detach().flatten().clone()

    denom = torch.linalg.vector_norm(g_global) * torch.linalg.vector_norm(g_local)
    return float(torch.dot(g_global, g_local) / denom) if denom > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=420)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--h", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        x, y, xt, yt = make_task(seed, d=args.d)
        row = {
            "seed": seed,
            "gradient_cosine_global_local": gradient_cosine(seed, x, y, args.d, args.h),
            "bp_test_mse": run_bp(seed, x, y, xt, yt, args.d, args.h, args.steps),
            "mf_simul_test_mse": run_mf_simul(seed, x, y, xt, yt, args.d, args.h, args.steps),
            "mf_seq_test_mse": run_mf_seq(seed, x, y, xt, yt, args.d, args.h, args.steps),
            "seq_global_test_mse": run_seq_global(seed, x, y, xt, yt, args.d, args.h, args.steps),
        }
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.out_dir / "w6_toy_per_seed.csv", index=False)

    methods = ["bp", "mf_simul", "mf_seq", "seq_global"]
    summary = {}
    for method in methods:
        vals = per_seed[f"{method}_test_mse"].to_numpy(float)
        summary[method] = {
            "mean_test_mse": float(np.mean(vals)),
            "median_test_mse": float(np.median(vals)),
            "sd_test_mse": float(np.std(vals, ddof=1)),
        }
    bp_mean = summary["bp"]["mean_test_mse"]
    for method in methods:
        summary[method]["ratio_to_bp_mean"] = (
            summary[method]["mean_test_mse"] / bp_mean if bp_mean > 0 else float("nan")
        )
    cos = per_seed.gradient_cosine_global_local.to_numpy(float)
    summary["gradient_cosine_global_local"] = {
        "mean": float(np.mean(cos)),
        "median": float(np.median(cos)),
        "sd": float(np.std(cos, ddof=1)),
    }
    (args.out_dir / "w6_toy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# W6 exploratory toy numerics",
        "",
        "**EXPLORATORY — NOT CONFIRMATORY.** No W6 preregistration exists yet. "
        "This is a CPU sanity check of the staging story on one synthetic deep-linear task; "
        "it must not be pooled with W1/W2/W3/W4 or cited as a hypothesis test.",
        "",
        f"Task: d={args.d}, hidden={args.h}, rank-2 teacher, {args.seeds} seeds, "
        f"{args.steps} Adam steps per method. Held-out MSE, lower is better.",
        "",
        "| method | mean MSE | median MSE | SD | ratio to BP mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        r = summary[method]
        lines.append(
            f"| {method} | {r['mean_test_mse']:.3e} | {r['median_test_mse']:.3e} | "
            f"{r['sd_test_mse']:.3e} | {r['ratio_to_bp_mean']:.3f}× |"
        )
    g = summary["gradient_cosine_global_local"]
    lines += [
        "",
        f"Gradient diagnostic at initialisation: cosine(global BP, first local head) "
        f"= {g['mean']:.3f} mean, {g['median']:.3f} median, SD {g['sd']:.3f}.",
        "",
        "Reading: this only checks whether a local/global gap is visible on a toy; it "
        "does not establish a mechanism. A confirmatory W6 addendum, if the author wants one, "
        "must specify the task family, seeds, metric, and thresholds before running it.",
    ]
    (args.out_dir / "w6_toy_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
