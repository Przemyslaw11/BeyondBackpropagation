#!/usr/bin/env python3
"""W3: hook-based byte accounting per pass and to stopping.

Counts tensor traffic through the exact execution graph of each arm on the
canonical 3x2000 CIFAR MLP:

  BP        one forward + one backward through W1..W3 and the output layer.
  MF-Simul  forward with detached inputs, local heads M0..M3, one backward.
  MF-Seq    stage M0 (full no-grad forward, as the code actually does, then M0
            head), then stages W1, W2, W3: frozen-prefix forward under no_grad,
            current W_k + ReLU + M_k head with backward.

Bytes are logical tensor traffic:
  forward   activation input read + output write + parameter read
  backward  grad_output read + grad_input write + parameter-grad accumulate
            (read+write of the parameter buffer)
Optimizer state/update traffic and the validation pass are excluded; both are
arm-independent per parameter/epoch and are stated in the summary. GB = 1e9 B.

Per pass = one epoch over 45,000 training samples (352 batches of 128).
Bytes-to-stopping = per-pass bytes x measured passes: JSON epochs_completed for
BP/MF-Simul, per-stage epochs parsed from each MF-Seq run's own stdout (W1_M1
missing line = 100-epoch cap; sum is asserted against JSON epochs).

MF-Seq canonical config is activation_cache=recompute because CIFAR random
crop/flip makes a frozen prefix's activations differ every epoch. The cached
prefix is therefore a counterfactual only (augmentation disabled); its one-time
fill and per-epoch read are reported separately.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from src.architectures.mf_mlp import MF_MLP

INPUT_DIM = 3072
HIDDEN = [2000, 2000, 2000]
BATCH = 128
TRAIN_SAMPLES = 45_000
N_BATCHES = int(np.ceil(TRAIN_SAMPLES / BATCH))
GB = 1e9


def t_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(values.mean() - half), float(values.mean() + half)


class Counter:
    def __init__(self) -> None:
        self.fwd_act = 0
        self.fwd_param = 0
        self.bwd_act = 0
        self.bwd_param = 0
        self.fwd_act_by_module: dict[str, int] = {}
        self.fwd_total_by_module: dict[str, int] = {}

    def forward_hook(self, name, module, inputs, output):
        in_b = sum(t.numel() * t.element_size() for t in inputs if torch.is_tensor(t))
        out_b = sum(
            t.numel() * t.element_size() for t in (output,) if torch.is_tensor(t)
        )
        p_b = sum(
            p.numel() * p.element_size() for p in module.parameters(recurse=False)
        )
        self.fwd_act += in_b + out_b
        self.fwd_param += p_b
        self.fwd_act_by_module[name] = (
            self.fwd_act_by_module.get(name, 0) + in_b + out_b
        )
        self.fwd_total_by_module[name] = (
            self.fwd_total_by_module.get(name, 0) + in_b + out_b + p_b
        )

    def backward_hook(self, name, module, grad_input, grad_output):
        gi = sum(t.numel() * t.element_size() for t in grad_input if torch.is_tensor(t))
        go = sum(
            t.numel() * t.element_size() for t in grad_output if torch.is_tensor(t)
        )
        self.bwd_act += gi + go
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            p_b = sum(
                p.numel() * p.element_size() for p in module.parameters(recurse=False)
            )
            self.bwd_param += 2 * p_b

    def total(self) -> int:
        return self.fwd_act + self.fwd_param + self.bwd_act + self.bwd_param

    def act_total(self) -> int:
        return self.fwd_act + self.bwd_act


def attach(model: MF_MLP) -> tuple[Counter, list]:
    counter = Counter()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
            handles.append(
                module.register_forward_hook(
                    lambda m, i, o, n=name: counter.forward_hook(n, m, i, o)
                )
            )
            handles.append(
                module.register_full_backward_hook(
                    lambda m, gi, go, n=name: counter.backward_hook(n, m, gi, go)
                )
            )
    return counter, handles


def detach(handles) -> None:
    for h in handles:
        h.remove()


def build(num_classes: int) -> MF_MLP:
    torch.manual_seed(0)
    return MF_MLP(INPUT_DIM, HIDDEN, num_classes, activation="relu")


def head_bytes(
    counter: Counter, a: torch.Tensor, m: torch.Tensor, num_classes: int
) -> None:
    a_b = a.numel() * a.element_size()
    m_b = m.numel() * m.element_size()
    log_b = BATCH * num_classes * 4
    counter.fwd_act += a_b + log_b
    counter.fwd_param += m_b
    counter.bwd_act += log_b + (a_b if a.requires_grad else 0)
    counter.bwd_param += 2 * m_b


def case_bp(num_classes: int) -> Counter:
    model = build(num_classes)
    counter, handles = attach(model)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, num_classes, (BATCH,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    detach(handles)
    return counter


def case_simul(num_classes: int) -> Counter:
    model = build(num_classes)
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith("output_layer"))
    counter, handles = attach(model)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, num_classes, (BATCH,))
    acts = [x]
    cur = x
    for i in range(len(HIDDEN)):
        z = model.layers[2 * i](cur.detach())
        a = model.layers[2 * i + 1](z)
        acts.append(a)
        cur = a
    total = F.cross_entropy(acts[-1] @ model.get_projection_matrix(len(HIDDEN)).t(), y)
    for i in range(len(HIDDEN)):
        total = total + 0.1 * F.cross_entropy(
            acts[i] @ model.get_projection_matrix(i).t(), y
        )
    for i in range(len(HIDDEN) + 1):
        head_bytes(counter, acts[i], model.get_projection_matrix(i), num_classes)
    total.backward()
    detach(handles)
    return counter


def case_m0(num_classes: int) -> Counter:
    model = build(num_classes)
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_projection_matrix(0).requires_grad_(True)
    counter, handles = attach(model)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, num_classes, (BATCH,))
    with torch.no_grad():
        acts = model.forward_with_intermediate_activations(x)
    loss = F.cross_entropy(acts[0] @ model.get_projection_matrix(0).t(), y)
    head_bytes(counter, acts[0], model.get_projection_matrix(0), num_classes)
    loss.backward()
    detach(handles)
    return counter


def case_stage(k: int, num_classes: int) -> Counter:
    """Stage that trains W_k and M_k; k = 1..L."""
    model = build(num_classes)
    for p in model.parameters():
        p.requires_grad_(False)
    model.layers[2 * (k - 1)].requires_grad_(True)
    model.get_projection_matrix(k).requires_grad_(True)
    counter, handles = attach(model)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, num_classes, (BATCH,))
    with torch.no_grad():
        prev = x
        for j in range(k - 1):
            prev = model.layers[2 * j + 1](model.layers[2 * j](prev))
    prev = prev.detach()
    z = model.layers[2 * (k - 1)](prev)
    a = model.layers[2 * (k - 1) + 1](z)
    loss = F.cross_entropy(a @ model.get_projection_matrix(k).t(), y)
    head_bytes(counter, a, model.get_projection_matrix(k), num_classes)
    loss.backward()
    detach(handles)
    return counter


def pack(counter: Counter, dataset: str, arm: str, stage: str) -> dict:
    return dict(
        dataset=dataset,
        arm=arm,
        stage=stage,
        batch_size=BATCH,
        n_batches=N_BATCHES,
        per_batch_total_gb=counter.total() / GB,
        fwd_act_gb=counter.fwd_act * N_BATCHES / GB,
        fwd_param_gb=counter.fwd_param * N_BATCHES / GB,
        bwd_act_gb=counter.bwd_act * N_BATCHES / GB,
        bwd_param_gb=counter.bwd_param * N_BATCHES / GB,
        total_gb=counter.total() * N_BATCHES / GB,
        act_only_gb=counter.act_total() * N_BATCHES / GB,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-per-seed", type=Path, required=True)
    parser.add_argument("--stage-anatomy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    g1 = pd.read_csv(args.g1_per_seed)
    anatomy = pd.read_csv(args.stage_anatomy)

    per_pass = []
    stage_counters: dict[tuple[str, int], Counter] = {}
    for dataset, num_classes in (("cifar10", 10), ("cifar100", 100)):
        stage_counters[(dataset, 0)] = case_m0(num_classes)
        for k in range(1, len(HIDDEN) + 1):
            stage_counters[(dataset, k)] = case_stage(k, num_classes)
        bp = case_bp(num_classes)
        sim = case_simul(num_classes)
        per_pass += [
            pack(bp, dataset, "bp", "full"),
            pack(sim, dataset, "mf_simul", "full"),
            pack(stage_counters[(dataset, 0)], dataset, "mf_seq", "M0"),
            *[
                pack(stage_counters[(dataset, k)], dataset, "mf_seq", f"W{k}")
                for k in range(1, len(HIDDEN) + 1)
            ],
        ]
    per_pass_df = pd.DataFrame(per_pass)
    per_pass_df.to_csv(args.out_dir / "w3_per_pass.csv", index=False)

    # per-seed bytes to stopping
    rows = []
    for dataset in ("cifar10", "cifar100"):
        for arm in ("bp", "mf_simul"):
            pp = per_pass_df[
                (per_pass_df.dataset == dataset) & (per_pass_df.arm == arm)
            ].iloc[0]
            for _, run in g1[(g1.dataset == dataset) & (g1.arm == arm)].iterrows():
                rows.append(
                    dict(
                        dataset=dataset,
                        arm=arm,
                        seed=int(run.seed),
                        passes=float(run.passes),
                        total_gb=pp.total_gb * run.passes,
                        act_only_gb=pp.act_only_gb * run.passes,
                    )
                )
        stages = per_pass_df[
            (per_pass_df.dataset == dataset) & (per_pass_df.arm == "mf_seq")
        ]
        stage_gb = stages.set_index("stage").total_gb.to_dict()
        stage_act = stages.set_index("stage").act_only_gb.to_dict()
        seq = anatomy[anatomy.experiment == f"mf_{dataset}_mlp_3x2000"]
        for _, run in seq.iterrows():
            w1 = float(run.w1_epochs) if pd.notna(run.w1_epochs) else 100.0
            epoch_map = {
                "M0": float(run.m0_epochs),
                "W1": w1,
                "W2": float(run.w2_epochs),
                "W3": float(run.w3_epochs),
            }
            total_passes = sum(epoch_map.values())
            g1_passes = float(
                g1[
                    (g1.dataset == dataset)
                    & (g1.arm == "mf_seq")
                    & (g1.seed == int(run.seed))
                ].passes.iloc[0]
            )
            if abs(total_passes - g1_passes) > 0.01:
                raise ValueError(
                    f"stage pass mismatch {dataset} seed{run.seed}: "
                    f"{total_passes} vs {g1_passes}"
                )
            rows.append(
                dict(
                    dataset=dataset,
                    arm="mf_seq",
                    seed=int(run.seed),
                    passes=total_passes,
                    total_gb=sum(stage_gb[s] * e for s, e in epoch_map.items()),
                    act_only_gb=sum(stage_act[s] * e for s, e in epoch_map.items()),
                )
            )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.out_dir / "w3_per_seed.csv", index=False)

    # arm summary
    summary = []
    for dataset in ("cifar10", "cifar100"):
        base = per_seed[(per_seed.dataset == dataset) & (per_seed.arm == "bp")]
        base_total = base.total_gb.mean()
        base_act = base.act_only_gb.mean()
        for arm in ("bp", "mf_simul", "mf_seq"):
            sub = per_seed[(per_seed.dataset == dataset) & (per_seed.arm == arm)]
            lo, hi = t_ci(sub.total_gb.to_numpy(float))
            alo, ahi = t_ci(sub.act_only_gb.to_numpy(float))
            summary.append(
                dict(
                    dataset=dataset,
                    arm=arm,
                    n=len(sub),
                    passes_mean=float(sub.passes.mean()),
                    total_gb_mean=float(sub.total_gb.mean()),
                    total_gb_lo=lo,
                    total_gb_hi=hi,
                    act_gb_mean=float(sub.act_only_gb.mean()),
                    act_gb_lo=alo,
                    act_gb_hi=ahi,
                    ratio_total_vs_bp=float(sub.total_gb.mean() / base_total),
                    ratio_act_vs_bp=float(sub.act_only_gb.mean() / base_act),
                )
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir / "w3_arm_summary.csv", index=False)

    # MF-Seq stage contributions
    stage_rows = []
    for dataset in ("cifar10", "cifar100"):
        stages = per_pass_df[
            (per_pass_df.dataset == dataset) & (per_pass_df.arm == "mf_seq")
        ].set_index("stage")
        seq = anatomy[anatomy.experiment == f"mf_{dataset}_mlp_3x2000"]
        epoch_means = {
            "M0": float(seq.m0_epochs.mean()),
            "W1": float(seq.w1_epochs.fillna(100).mean()),
            "W2": float(seq.w2_epochs.mean()),
            "W3": float(seq.w3_epochs.mean()),
        }
        total_contrib = sum(
            stages.loc[s, "total_gb"] * e for s, e in epoch_means.items()
        )
        total_act = sum(
            stages.loc[s, "act_only_gb"] * e for s, e in epoch_means.items()
        )
        for s in ("M0", "W1", "W2", "W3"):
            stage_rows.append(
                dict(
                    dataset=dataset,
                    stage=s,
                    epochs_mean=epoch_means[s],
                    per_pass_total_gb=float(stages.loc[s, "total_gb"]),
                    per_pass_act_gb=float(stages.loc[s, "act_only_gb"]),
                    contribution_gb=float(stages.loc[s, "total_gb"] * epoch_means[s]),
                    share_pct=float(
                        100 * stages.loc[s, "total_gb"] * epoch_means[s] / total_contrib
                    ),
                    act_contribution_gb=float(
                        stages.loc[s, "act_only_gb"] * epoch_means[s]
                    ),
                    act_share_pct=float(
                        100 * stages.loc[s, "act_only_gb"] * epoch_means[s] / total_act
                    ),
                )
            )
    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(args.out_dir / "w3_stage_contributions.csv", index=False)

    # cached-prefix counterfactual (augmentation disabled)
    cache_rows = []
    for dataset in ("cifar10", "cifar100"):
        prefix_total = {1: 0.0, 2: 0.0, 3: 0.0}
        prefix_act = {1: 0.0, 2: 0.0, 3: 0.0}
        for k in (2, 3):
            c = stage_counters[(dataset, k)]
            for j in range(k - 1):
                for name in (f"layers.{2 * j}", f"layers.{2 * j + 1}"):
                    prefix_total[k] += (
                        c.fwd_total_by_module.get(name, 0) * N_BATCHES / GB
                    )
                    prefix_act[k] += c.fwd_act_by_module.get(name, 0) * N_BATCHES / GB
        seq = anatomy[anatomy.experiment == f"mf_{dataset}_mlp_3x2000"]
        primary_total = summary_df[
            (summary_df.dataset == dataset) & (summary_df.arm == "mf_seq")
        ].total_gb_mean.iloc[0]
        primary_act = summary_df[
            (summary_df.dataset == dataset) & (summary_df.arm == "mf_seq")
        ].act_gb_mean.iloc[0]
        # Start from the per-seed total (not the mean): the loop adds one
        # correction per run, then divides by n.
        cached_total = primary_total * len(seq)
        cached_act = primary_act * len(seq)
        write_a_gb = TRAIN_SAMPLES * HIDDEN[0] * 4 / GB
        for _, run in seq.iterrows():
            w1 = float(run.w1_epochs) if pd.notna(run.w1_epochs) else 100.0
            emap = {"W1": w1, "W2": float(run.w2_epochs), "W3": float(run.w3_epochs)}
            for k in (2, 3):
                e = emap[f"W{k}"]
                cached_total += -prefix_total[k] * e + prefix_total[k] + write_a_gb
                cached_act += -prefix_act[k] * e + prefix_act[k] + write_a_gb
        cached_total /= len(seq)
        cached_act /= len(seq)
        cache_rows.append(
            dict(
                dataset=dataset,
                mfseq_total_gb=primary_total,
                cached_total_gb=cached_total,
                total_savings_pct=100 * (primary_total - cached_total) / primary_total,
                mfseq_act_gb=primary_act,
                cached_act_gb=cached_act,
                act_savings_pct=100 * (primary_act - cached_act) / primary_act,
            )
        )
    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_csv(args.out_dir / "w3_cache_counterfactual.csv", index=False)

    # markdown
    lines = ["# W3 — hook-based byte accounting", ""]
    lines.append(
        f"Canonical 3×2000 CIFAR MLP, batch {BATCH}, {N_BATCHES} batches/pass "
        f"({TRAIN_SAMPLES:,} training samples). Logical tensor traffic: forward "
        "activation read+write and parameter read; backward grad read/write and "
        "parameter-grad accumulate. Optimizer state/update and the validation pass "
        "are excluded. GB = 1e9 bytes."
    )
    lines.append("")
    lines.append("## Per-pass (one epoch) traffic, GB")
    lines.append("")
    lines.append(
        "| dataset | arm | stage | forward act | forward param | backward act | backward param | total | act-only |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in per_pass_df.iterrows():
        lines.append(
            f"| {r.dataset} | {r.arm} | {r.stage} | {r.fwd_act_gb:.3f} | "
            f"{r.fwd_param_gb:.3f} | {r.bwd_act_gb:.3f} | {r.bwd_param_gb:.3f} | "
            f"**{r.total_gb:.3f}** | {r.act_only_gb:.3f} |"
        )
    lines.append("")
    lines.append("## Bytes to stopping (measured passes)")
    lines.append("")
    lines.append(
        "| dataset | arm | n | passes | total GB [95% CI] | ratio vs BP | act-only GB [95% CI] | act ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r.dataset} | {r.arm} | {r.n} | {r.passes_mean:.1f} | "
            f"{r.total_gb_mean:.1f} [{r.total_gb_lo:.1f}, {r.total_gb_hi:.1f}] | "
            f"{r.ratio_total_vs_bp:.2f}× | "
            f"{r.act_gb_mean:.1f} [{r.act_gb_lo:.1f}, {r.act_gb_hi:.1f}] | "
            f"{r.ratio_act_vs_bp:.2f}× |"
        )
    lines.append("")
    lines.append("## MF-Seq stage contributions")
    lines.append("")
    lines.append(
        "| dataset | stage | mean epochs | per-pass GB | contribution GB | share | act-only share |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in stage_df.iterrows():
        lines.append(
            f"| {r.dataset} | {r.stage} | {r.epochs_mean:.1f} | {r.per_pass_total_gb:.3f} | "
            f"{r.contribution_gb:.1f} | {r.share_pct:.1f}% | {r.act_share_pct:.1f}% |"
        )
    lines.append("")
    lines.append("## Cached-prefix counterfactual")
    lines.append("")
    lines.append(
        "Canonical MF-Seq uses `activation_cache: recompute`: CIFAR random crop/flip "
        "changes the input every epoch, so a frozen prefix's activations cannot be "
        "reused and the code raises on a cache strategy under stochastic transforms. "
        "The table below is the bound if augmentation were removed and the prefix "
        "activation were cached: one fill per stage, then a cached read each epoch."
    )
    lines.append("")
    lines.append(
        "| dataset | MF-Seq total GB | cached total GB | saving | act-only GB | cached act GB | act saving |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in cache_df.iterrows():
        lines.append(
            f"| {r.dataset} | {r.mfseq_total_gb:.0f} | {r.cached_total_gb:.0f} | "
            f"{r.total_savings_pct:.1f}% | {r.mfseq_act_gb:.0f} | {r.cached_act_gb:.0f} | "
            f"{r.act_savings_pct:.1f}% |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "BP and MF-Simul pay a full backward through every layer; MF-Seq pays only "
        "the current layer's backward but recomputes the frozen prefix every epoch. "
        "The M0 stage is pure overhead in both senses: no prediction reads M0, and "
        "the code still runs a full no-grad forward through W1..W3 every batch "
        "(train_mf_matrix_only calls forward_with_intermediate_activations before "
        "taking a_0). Its share is therefore a lower bound on the dead-stage cost."
    )
    (args.out_dir / "w3_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
