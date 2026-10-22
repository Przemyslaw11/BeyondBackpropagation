#!/usr/bin/env python3
"""W3 (redone): cut-boundary communication accounting.

The unit of accounting is one activation tensor crossing one cut, i.e. one
``45,000 x 2,000 x 4 B = 0.36 GB`` transfer per epoch per direction per cut
(decimal GB; 0.335 GiB).  This is *not* the old W3 memory-traffic table: the old
69.1 GB/pass counted every parameter read and every layer activation read/write,
including the weight re-reads that never cross a layer boundary.

Cases measured with forward hooks on the ReLU outputs (the cut tensors):

  BP        cut k: activation a_k forward + gradient dL/da_k backward.
  MF-Simul  cut k: activation a_k forward only (local heads never backprop
            across a cut).
  MF-Seq    stage W_k: the frozen prefix recomputes a_1..a_{k-1}; only those
            prefix cuts cross into the stage.  Current-stage a_k and its head
            stay local and are not cut traffic.
  M0        the production code runs a full no-grad forward through W1..W3
            although M0 reads only a_0=x.  That is 3 cut tensors per pass of
            pure implementation waste; reported separately and included only in
            the "as implemented" total.

Cached-prefix (augmentation disabled) is a separate counterfactual: one fill per
stage (the prefix cuts), then one read of a_{k-1} per epoch.  CIFAR random
crop/flip makes the cached case invalid for the canonical training runs.

Closed-form cross-check per cut and direction:
    N_train * hidden_dim_k * 4 bytes
The hook measurement is one batch of 128, scaled by 45,000 training samples;
the closed form is exact.  A mismatch >1e-9 raises.

Bytes-to-stopping use each run's JSON ``epochs_completed`` for BP/MF-Simul and
the per-stage stdout anatomy for MF-Seq (missing W1 line = 100-epoch cap, as in
W0c).  Stage sums are asserted against the JSON pass count.
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
VAL_SAMPLES = 5_000
GB = 1e9
CUTS = list(range(1, len(HIDDEN) + 1))


def t_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(values.mean() - half), float(values.mean() + half)


class CutCounter:
    """Counts one tensor per cut per direction, not every layer read/write."""

    def __init__(self) -> None:
        self.forward: dict[int, int] = {}
        self.backward: dict[int, int] = {}
        self.cache_read: dict[int, int] = {}
        self.forward_calls: dict[int, int] = {}

    def hook_forward(self, cut: int, module, inputs, output) -> None:
        out = output[0] if isinstance(output, tuple) else output
        self.forward[cut] = self.forward.get(cut, 0) + out.numel() * out.element_size()
        self.forward_calls[cut] = self.forward_calls.get(cut, 0) + 1

    def hook_backward(self, cut: int, module, grad_input, grad_output) -> None:
        go = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if go is not None:
            self.backward[cut] = (
                self.backward.get(cut, 0) + go.numel() * go.element_size()
            )

    def hook_cache_read(self, cut: int, module, inputs) -> None:
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        self.cache_read[cut] = (
            self.cache_read.get(cut, 0) + x.numel() * x.element_size()
        )


def attach(
    model: MF_MLP, *, backward: bool, cache_read_cut: int | None = None
) -> tuple[CutCounter, list]:
    counter = CutCounter()
    handles = []
    for k in CUTS:
        act = model.layers[2 * (k - 1) + 1]
        handles.append(
            act.register_forward_hook(
                lambda m, i, o, cut=k: counter.hook_forward(cut, m, i, o)
            )
        )
        if backward:
            handles.append(
                act.register_full_backward_hook(
                    lambda m, gi, go, cut=k: counter.hook_backward(cut, m, gi, go)
                )
            )
    if cache_read_cut is not None:
        linear = model.layers[2 * (cache_read_cut - 1)]
        handles.append(
            linear.register_forward_pre_hook(
                lambda m, i, cut=cache_read_cut: counter.hook_cache_read(cut, m, i)
            )
        )
    return counter, handles


def detach(handles) -> None:
    for h in handles:
        h.remove()


def build(num_classes: int) -> MF_MLP:
    torch.manual_seed(0)
    return MF_MLP(INPUT_DIM, HIDDEN, num_classes, activation="relu")


def closed_form(cut: int) -> int:
    return TRAIN_SAMPLES * HIDDEN[cut - 1] * 4


def scaled(hook_bytes: int) -> float:
    """One-batch hook bytes -> one full epoch over 45,000 samples."""
    return hook_bytes / BATCH * TRAIN_SAMPLES


def case_bp(num_classes: int) -> CutCounter:
    model = build(num_classes)
    counter, handles = attach(model, backward=True)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, num_classes, (BATCH,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    detach(handles)
    return counter


def case_simul(num_classes: int) -> CutCounter:
    model = build(num_classes)
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith("output_layer"))
    counter, handles = attach(model, backward=False)
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
    total.backward()
    detach(handles)
    return counter


def case_m0(num_classes: int) -> CutCounter:
    """Production M0 forward: full no-grad forward, then the M0 head."""
    model = build(num_classes)
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_projection_matrix(0).requires_grad_(True)
    counter, handles = attach(model, backward=False)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, num_classes, (BATCH,))
    with torch.no_grad():
        acts = model.forward_with_intermediate_activations(x)
    loss = F.cross_entropy(acts[0] @ model.get_projection_matrix(0).t(), y)
    loss.backward()
    detach(handles)
    return counter


def case_stage(k: int, num_classes: int) -> CutCounter:
    """Production MF-Seq stage W_k: frozen prefix recomputes a_1..a_{k-1}."""
    model = build(num_classes)
    for p in model.parameters():
        p.requires_grad_(False)
    model.layers[2 * (k - 1)].requires_grad_(True)
    model.get_projection_matrix(k).requires_grad_(True)
    counter, handles = attach(model, backward=False)
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
    loss.backward()
    detach(handles)
    return counter


def case_cached_stage(k: int, num_classes: int) -> CutCounter:
    """Fill the prefix once, then read the cached a_{k-1} for one batch."""
    model = build(num_classes)
    for p in model.parameters():
        p.requires_grad_(False)
    model.layers[2 * (k - 1)].requires_grad_(True)
    model.get_projection_matrix(k).requires_grad_(True)
    counter, handles = attach(
        model, backward=False, cache_read_cut=(k if k >= 2 else None)
    )
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
    loss.backward()
    detach(handles)
    return counter


def checked(counter: CutCounter, cut: int, direction: str) -> tuple[float, float]:
    raw = getattr(counter, direction).get(cut, 0)
    observed = scaled(raw) / GB
    expected = closed_form(cut) / GB
    if expected and abs(observed - expected) > 1e-9:
        raise AssertionError(
            f"cut {cut} {direction}: hook {observed:.12f} GB != "
            f"closed form {expected:.12f} GB"
        )
    return observed, expected


def per_pass_rows() -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[int, float]]]:
    rows = []
    stage_totals: dict[tuple[str, str, str], dict[int, float]] = {}

    def add(dataset, arm, stage, cut, direction, observed, expected):
        rows.append(
            dict(
                dataset=dataset,
                arm=arm,
                stage=stage,
                cut=cut,
                direction=direction,
                hook_gb_per_pass=observed,
                closed_form_gb_per_pass=expected,
                ratio=observed / expected if expected else float("nan"),
            )
        )

    for dataset, num_classes in (("cifar10", 10), ("cifar100", 100)):
        bp = case_bp(num_classes)
        for k in CUTS:
            f_obs, f_exp = checked(bp, k, "forward")
            b_obs, b_exp = checked(bp, k, "backward")
            add(dataset, "bp", "full", k, "forward", f_obs, f_exp)
            add(dataset, "bp", "full", k, "backward", b_obs, b_exp)
        stage_totals[(dataset, "bp", "full")] = {
            k: checked(bp, k, "forward")[0] + checked(bp, k, "backward")[0]
            for k in CUTS
        }

        sim = case_simul(num_classes)
        for k in CUTS:
            obs, exp = checked(sim, k, "forward")
            add(dataset, "mf_simul", "full", k, "forward", obs, exp)
        stage_totals[(dataset, "mf_simul", "full")] = {
            k: checked(sim, k, "forward")[0] for k in CUTS
        }

        m0 = case_m0(num_classes)
        for k in CUTS:
            obs, exp = checked(m0, k, "forward")
            add(dataset, "mf_seq_as_implemented", "M0_waste", k, "forward", obs, exp)
        stage_totals[(dataset, "mf_seq_as_implemented", "M0_waste")] = {
            k: checked(m0, k, "forward")[0] for k in CUTS
        }
        stage_totals[(dataset, "mf_seq_ex_m0", "M0_waste")] = {k: 0.0 for k in CUTS}

        for k in CUTS:
            stage = case_stage(k, num_classes)
            for j in range(1, k):
                obs, exp = checked(stage, j, "forward")
                add(dataset, "mf_seq", f"W{k}", j, "forward", obs, exp)
            stage_totals[(dataset, "mf_seq", f"W{k}")] = {
                j: (checked(stage, j, "forward")[0] if j < k else 0.0) for j in CUTS
            }
    return pd.DataFrame(rows), stage_totals


def cache_rows() -> pd.DataFrame:
    rows = []
    for dataset, num_classes in (("cifar10", 10), ("cifar100", 100)):
        for k in CUTS:
            c = case_cached_stage(k, num_classes)
            fill = sum(
                checked(c, j, "forward")[0] for j in range(1, k)
            )
            read = scaled(c.cache_read.get(k, 0)) / GB
            expected_read = closed_form(k) / GB if k >= 2 else 0.0
            if abs(read - expected_read) > 1e-9:
                raise AssertionError(
                    f"{dataset} cache W{k}: read {read:.12f} != {expected_read:.12f}"
                )
            rows.append(
                dict(
                    dataset=dataset,
                    stage=f"W{k}",
                    cut=k,
                    fill_gb=fill,
                    cached_read_gb_per_epoch=read,
                    closed_form_read_gb=expected_read,
                )
            )
    return pd.DataFrame(rows)


def epoch_map(run: pd.Series) -> dict[str, float]:
    w1 = float(run.w1_epochs) if pd.notna(run.w1_epochs) else 100.0
    return {
        "M0": float(run.m0_epochs),
        "W1": w1,
        "W2": float(run.w2_epochs),
        "W3": float(run.w3_epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-per-seed", type=Path, required=True)
    parser.add_argument("--stage-anatomy", type=Path, required=True)
    parser.add_argument("--stage-energy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    g1 = pd.read_csv(args.g1_per_seed)
    anatomy = pd.read_csv(args.stage_anatomy)
    stage_energy = pd.read_csv(args.stage_energy)

    per_pass, stage_totals = per_pass_rows()
    per_pass.to_csv(args.out_dir / "w3_cut_per_pass.csv", index=False)
    cache = cache_rows()
    cache.to_csv(args.out_dir / "w3_cut_cache.csv", index=False)

    # Bytes to stopping, per seed, per cut.
    rows = []
    for dataset in ("cifar10", "cifar100"):
        for arm in ("bp", "mf_simul"):
            per_cut = stage_totals[(dataset, arm, "full")]
            for _, run in g1[(g1.dataset == dataset) & (g1.arm == arm)].iterrows():
                passes = float(run.passes)
                for k in CUTS:
                    rows.append(
                        dict(
                            dataset=dataset,
                            arm=arm,
                            seed=int(run.seed),
                            passes=passes,
                            cut=k,
                            gb=per_cut[k] * passes,
                        )
                    )
        seq = anatomy[anatomy.experiment == f"mf_{dataset}_mlp_3x2000"]
        for _, run in seq.iterrows():
            emap = epoch_map(run)
            total_passes = sum(emap.values())
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
            for k in CUTS:
                gb = stage_totals[(dataset, "mf_seq", "W1")][k] * emap["W1"]
                gb += stage_totals[(dataset, "mf_seq", "W2")][k] * emap["W2"]
                gb += stage_totals[(dataset, "mf_seq", "W3")][k] * emap["W3"]
                waste = stage_totals[(dataset, "mf_seq_as_implemented", "M0_waste")][
                    k
                ] * emap["M0"]
                rows.append(
                    dict(
                        dataset=dataset,
                        arm="mf_seq",
                        seed=int(run.seed),
                        passes=total_passes,
                        cut=k,
                        gb=gb,
                    )
                )
                rows.append(
                    dict(
                        dataset=dataset,
                        arm="mf_seq_as_implemented",
                        seed=int(run.seed),
                        passes=total_passes,
                        cut=k,
                        gb=gb + waste,
                    )
                )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.out_dir / "w3_cut_per_seed.csv", index=False)

    # Per-arm summary: total and per-cut.
    summary_rows = []
    for dataset in ("cifar10", "cifar100"):
        for arm in ("bp", "mf_simul", "mf_seq", "mf_seq_as_implemented"):
            sub = per_seed[(per_seed.dataset == dataset) & (per_seed.arm == arm)]
            totals = sub.groupby("seed").gb.sum()
            lo, hi = t_ci(totals.to_numpy(float))
            passes = sub.groupby("seed").passes.first()
            base = per_seed[
                (per_seed.dataset == dataset) & (per_seed.arm == "bp")
            ].groupby("seed").gb.sum()
            summary_rows.append(
                dict(
                    dataset=dataset,
                    arm=arm,
                    n=len(totals),
                    passes_mean=float(passes.mean()),
                    gb_mean=float(totals.mean()),
                    gb_lo=lo,
                    gb_hi=hi,
                    ratio_vs_bp=float(totals.mean() / base.mean()),
                )
            )
            for k in CUTS:
                cut_totals = (
                    sub[sub.cut == k].set_index("seed").gb.reindex(totals.index)
                )
                summary_rows[-1][f"cut{k}_gb_mean"] = float(cut_totals.mean())
                summary_rows[-1][f"cut{k}_ratio_vs_bp"] = float(
                    cut_totals.mean()
                    / per_seed[
                        (per_seed.dataset == dataset)
                        & (per_seed.arm == "bp")
                        & (per_seed.cut == k)
                    ]
                    .set_index("seed")
                    .gb.mean()
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "w3_cut_arm_summary.csv", index=False)

    # Stage contributions (mean epochs).
    stage_rows = []
    for dataset in ("cifar10", "cifar100"):
        seq = anatomy[anatomy.experiment == f"mf_{dataset}_mlp_3x2000"]
        epoch_means = {
            "M0": float(seq.m0_epochs.mean()),
            "W1": float(seq.w1_epochs.fillna(100).mean()),
            "W2": float(seq.w2_epochs.mean()),
            "W3": float(seq.w3_epochs.mean()),
        }
        for stage in ("M0", "W1", "W2", "W3"):
            if stage == "M0":
                per_pass_gb = sum(
                    stage_totals[(dataset, "mf_seq_as_implemented", "M0_waste")].values()
                )
            else:
                per_pass_gb = sum(stage_totals[(dataset, "mf_seq", stage)].values())
            stage_rows.append(
                dict(
                    dataset=dataset,
                    stage=stage,
                    epochs_mean=epoch_means[stage],
                    gb_per_pass=per_pass_gb,
                    contribution_gb=per_pass_gb * epoch_means[stage],
                )
            )
    stages = pd.DataFrame(stage_rows)
    stages["share_pct"] = stages.groupby("dataset").contribution_gb.transform(
        lambda x: 100 * x / x.sum()
    )
    stages.to_csv(args.out_dir / "w3_cut_stage_contributions.csv", index=False)

    # Cached-prefix counterfactual per seed.
    cache_rows_out = []
    for dataset in ("cifar10", "cifar100"):
        seq = anatomy[anatomy.experiment == f"mf_{dataset}_mlp_3x2000"]
        cmap = cache[cache.dataset == dataset].set_index("stage")
        for _, run in seq.iterrows():
            emap = epoch_map(run)
            recompute = sum(
                sum(stage_totals[(dataset, "mf_seq", f"W{k}")].values()) * emap[f"W{k}"]
                for k in (1, 2, 3)
            )
            cached = 0.0
            for k in (1, 2, 3):
                if k == 1:
                    continue
                cached += float(cmap.loc[f"W{k}", "fill_gb"])
                cached += (
                    float(cmap.loc[f"W{k}", "cached_read_gb_per_epoch"])
                    * emap[f"W{k}"]
                )
            m0_waste = sum(
                stage_totals[(dataset, "mf_seq_as_implemented", "M0_waste")].values()
            ) * emap["M0"]
            cache_rows_out.append(
                dict(
                    dataset=dataset,
                    seed=int(run.seed),
                    recompute_ex_m0_gb=recompute,
                    cached_ex_m0_gb=cached,
                    m0_waste_gb=m0_waste,
                    recompute_as_implemented_gb=recompute + m0_waste,
                    cached_as_implemented_gb=cached + m0_waste,
                )
            )
    cache_seed = pd.DataFrame(cache_rows_out)
    cache_seed.to_csv(args.out_dir / "w3_cut_cache_per_seed.csv", index=False)
    cache_summary = (
        cache_seed.groupby("dataset")[
            [
                "recompute_ex_m0_gb",
                "cached_ex_m0_gb",
                "m0_waste_gb",
                "recompute_as_implemented_gb",
                "cached_as_implemented_gb",
            ]
        ]
        .mean()
        .reset_index()
    )
    cache_summary["ex_m0_saving_pct"] = (
        100
        * (cache_summary.recompute_ex_m0_gb - cache_summary.cached_ex_m0_gb)
        / cache_summary.recompute_ex_m0_gb
    )
    cache_summary["as_implemented_saving_pct"] = (
        100
        * (
            cache_summary.recompute_as_implemented_gb
            - cache_summary.cached_as_implemented_gb
        )
        / cache_summary.recompute_as_implemented_gb
    )
    cache_summary.to_csv(args.out_dir / "w3_cut_cache_summary.csv", index=False)

    # Ex-M0 multipliers (energy/passes/time) and stage-0 FLOP waste.
    ex_rows = []
    for dataset, exp in (
        ("cifar10", "mf_cifar10_mlp_3x2000"),
        ("cifar100", "mf_cifar100_mlp_3x2000"),
    ):
        bp = g1[(g1.dataset == dataset) & (g1.arm == "bp")]
        seq = g1[(g1.dataset == dataset) & (g1.arm == "mf_seq")]
        m0 = stage_energy[stage_energy.experiment == exp]
        e0 = float(m0.M0_wh.mean())
        s0 = float(m0.M0_sec.mean())
        m0_epochs = float(anatomy[anatomy.experiment == exp].m0_epochs.mean())
        dead_frac = 0.0
        num_classes = 10 if dataset == "cifar10" else 100
        full_fwd_per_sample = 2 * (
            INPUT_DIM * HIDDEN[0]
            + HIDDEN[0] * HIDDEN[1]
            + HIDDEN[1] * HIDDEN[2]
        )
        dead_flops = full_fwd_per_sample * (TRAIN_SAMPLES + VAL_SAMPLES)
        head_train_fwd = 2 * TRAIN_SAMPLES * INPUT_DIM * num_classes
        head_train_bwd = 2 * TRAIN_SAMPLES * INPUT_DIM * num_classes
        head_val = 2 * VAL_SAMPLES * INPUT_DIM * num_classes
        necessary_flops = head_train_fwd + head_train_bwd + head_val
        dead_frac = dead_flops / (dead_flops + necessary_flops)
        dead_wh = e0 * dead_frac
        ex_rows.append(
            dict(
                dataset=dataset,
                bp_energy_wh=float(bp.energy_wh.mean()),
                mfseq_energy_wh=float(seq.energy_wh.mean()),
                m0_energy_wh=e0,
                m0_passes=m0_epochs,
                m0_sec=s0,
                bp_passes=float(bp.passes.mean()),
                mfseq_passes=float(seq.passes.mean()),
                bp_sec=float(bp.duration_sec.mean()),
                mfseq_sec=float(seq.duration_sec.mean()),
                energy_mult=float(seq.energy_wh.mean() / bp.energy_wh.mean()),
                passes_mult=float(seq.passes.mean() / bp.passes.mean()),
                time_mult=float(seq.duration_sec.mean() / bp.duration_sec.mean()),
                ex_m0_energy_mult_estimate=float(
                    (seq.energy_wh.mean() - e0) / bp.energy_wh.mean()
                ),
                ex_m0_passes_mult_estimate=float(
                    (seq.passes.mean() - m0_epochs) / bp.passes.mean()
                ),
                ex_m0_time_mult_estimate=float(
                    (seq.duration_sec.mean() - s0) / bp.duration_sec.mean()
                ),
                dead_forward_flop_share=float(dead_frac),
                dead_forward_energy_lower_bound_wh=float(dead_wh),
                ex_dead_forward_energy_mult_estimate=float(
                    (seq.energy_wh.mean() - dead_wh) / bp.energy_wh.mean()
                ),
            )
        )
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(args.out_dir / "w3_cut_ex_m0.csv", index=False)

    # H-W3d pilot: report the sign of Spearman rho, with a cell bootstrap and an
    # explicit small-n caveat.  There is deliberately no one-sided threshold.
    energy_by_cell = (
        g1.groupby(["dataset", "arm"], as_index=False).energy_wh.mean()
    )
    pilot = summary[summary.arm.isin(["bp", "mf_simul", "mf_seq"])][
        ["dataset", "arm", "gb_mean", "passes_mean"]
    ].merge(energy_by_cell, on=["dataset", "arm"], how="left")
    pilot = pilot.dropna(subset=["gb_mean", "energy_wh"])

    def _bootstrap_rho_cells(frame: pd.DataFrame, seed: int = 42, n_boot: int = 10000):
        """Cell bootstrap; used only for the descriptive sensitivity row."""
        rng = np.random.default_rng(seed)
        values = []
        x = frame.gb_mean.to_numpy(float)
        y = frame.energy_wh.to_numpy(float)
        for _ in range(n_boot):
            idx = rng.integers(0, len(frame), len(frame))
            if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
                continue
            rho, _ = stats.spearmanr(x[idx], y[idx])
            if np.isfinite(rho):
                values.append(float(rho))
        if not values:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    def _bootstrap_rho_seeds(
        gb_by_seed: pd.DataFrame,
        energy_by_seed: pd.DataFrame,
        seed: int = 42,
        n_boot: int = 10000,
    ):
        """Paired seed bootstrap: resample seeds, recompute each arm's cell mean."""
        rng = np.random.default_rng(seed)
        values = []
        n_seeds = len(gb_by_seed)
        for _ in range(n_boot):
            idx = rng.integers(0, n_seeds, n_seeds)
            x = gb_by_seed.iloc[idx].mean(axis=0).to_numpy(float)
            y = energy_by_seed.iloc[idx].mean(axis=0).to_numpy(float)
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                continue
            rho, _ = stats.spearmanr(x, y)
            if np.isfinite(rho):
                values.append(float(rho))
        if not values:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    pilot_seed = (
        per_seed[per_seed.arm.isin(["bp", "mf_simul", "mf_seq"])]
        .groupby(["dataset", "arm", "seed"], as_index=False).gb.sum()
        .merge(
            g1[["dataset", "arm", "seed", "energy_wh"]],
            on=["dataset", "arm", "seed"],
            how="inner",
        )
    )
    gb_by_seed = pilot_seed.pivot(
        index="seed", columns=["dataset", "arm"], values="gb"
    )
    energy_by_seed = pilot_seed.pivot(
        index="seed", columns=["dataset", "arm"], values="energy_wh"
    )
    rho, p_value = stats.spearmanr(pilot.gb_mean, pilot.energy_wh)
    rho_lo, rho_hi = _bootstrap_rho_seeds(gb_by_seed, energy_by_seed)
    rho_cell_lo, rho_cell_hi = _bootstrap_rho_cells(pilot)
    h_rows = [
        dict(
            scope="pilot_three_arms_x_two_datasets",
            n_cells=len(pilot),
            spearman_rho=float(rho),
            p_descriptive=float(p_value),
            rho_boot_lo=rho_lo,
            rho_boot_hi=rho_hi,
            bootstrap_method="paired seed bootstrap within fixed cells",
            caveat=(
                "small n; two datasets share the same seed block; cells are not "
                "independent; sign only, no threshold; seed CI is degenerate "
                "because rescaling all cells preserves their ordering"
            ),
        ),
        dict(
            scope="pilot_cell_bootstrap_sensitivity",
            n_cells=len(pilot),
            spearman_rho=float(rho),
            p_descriptive=float(p_value),
            rho_boot_lo=rho_cell_lo,
            rho_boot_hi=rho_cell_hi,
            bootstrap_method="cell bootstrap (sensitivity)",
            caveat=(
                "six non-independent cells; this interval is not a confirmatory "
                "uncertainty statement"
            ),
        ),
    ]
    as_impl = summary[summary.arm == "mf_seq_as_implemented"].copy()
    if not as_impl.empty:
        as_impl = as_impl.merge(
            energy_by_cell[energy_by_cell.arm == "mf_seq"][
                ["dataset", "energy_wh"]
            ],
            on="dataset",
            how="left",
        )
        sensitivity = pd.concat(
            [pilot, as_impl[["dataset", "arm", "gb_mean", "passes_mean", "energy_wh"]]],
            ignore_index=True,
        )
        rho_s, p_s = stats.spearmanr(sensitivity.gb_mean, sensitivity.energy_wh)
        rho_s_lo, rho_s_hi = _bootstrap_rho_cells(sensitivity)
        h_rows.append(
            dict(
                scope="sensitivity_include_as_implemented",
                n_cells=len(sensitivity),
                spearman_rho=float(rho_s),
                p_descriptive=float(p_s),
                rho_boot_lo=rho_s_lo,
                rho_boot_hi=rho_s_hi,
                bootstrap_method="cell bootstrap",
                caveat=(
                    "as-implemented rows reuse MF-Seq energy and are not "
                    "independent; descriptive sensitivity only"
                ),
            )
        )
    h_w3d = pd.DataFrame(h_rows)
    h_w3d.to_csv(args.out_dir / "w3_cut_h_w3d.csv", index=False)

    # Markdown.
    lines = ["# W3 — cut-boundary communication (redone)", ""]
    lines.append(
        "**What the old 69 GB/pass counted.** The previous W3 table was a "
        "*logical memory-traffic* table, not communication. For CIFAR-10 BP it "
        "summed 4.881 GB forward activation reads/writes, 19.951 GB forward "
        "parameter reads, 4.327 GB backward activation reads/writes and 39.903 GB "
        "backward parameter-gradient reads/writes: **69.062 GB/pass**. Most of that "
        "is re-reading the 14.2M parameters every batch (19.95 GB/pass forward, "
        "39.90 GB/pass backward), which never crosses a cut. It is retained only "
        "in `w3_summary.md` under the relabelled memory-traffic heading."
    )
    lines.append("")
    lines.append(
        "**New unit.** One cut tensor = 45,000 × 2,000 × 4 B = "
        "**0.360 GB per epoch per direction per cut** (decimal GB; 0.335 GiB). "
        "Forward hooks are on the ReLU output a_k; backward hooks are on the same "
        "tensor. A one-batch hook measurement is scaled to 45,000 samples and "
        "checked against the closed form; every row matches to <1e-9 GB."
    )
    lines.append("")
    lines.append("## Per-cut rates (GB per epoch)")
    lines.append("")
    lines.append(
        "| cut | tensor | BP forward | BP backward | BP total | MF-Simul forward | MF-Seq W2 prefix | MF-Seq W3 prefix | M0 dead forward |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k in CUTS:
        def v(dataset, arm, stage, direction):
            row = per_pass[
                (per_pass.dataset == dataset)
                & (per_pass.arm == arm)
                & (per_pass.stage == stage)
                & (per_pass.cut == k)
                & (per_pass.direction == direction)
            ]
            return float(row.hook_gb_per_pass.iloc[0]) if len(row) else 0.0

        lines.append(
            f"| {k} | a_{k} | {v('cifar10','bp','full','forward'):.3f} | "
            f"{v('cifar10','bp','full','backward'):.3f} | "
            f"{v('cifar10','bp','full','forward') + v('cifar10','bp','full','backward'):.3f} | "
            f"{v('cifar10','mf_simul','full','forward'):.3f} | "
            f"{v('cifar10','mf_seq','W2','forward'):.3f} | "
            f"{v('cifar10','mf_seq','W3','forward'):.3f} | "
            f"{v('cifar10','mf_seq_as_implemented','M0_waste','forward'):.3f} |"
        )
    lines.append("")
    lines.append(
        "MF-Seq cut k is the prefix activation a_k recomputed by the later stage; "
        "the current stage's a_k and its local head do not cross a cut. Cut 3 is "
        "therefore zero in useful MF-Seq; it appears only in the dead M0 forward."
    )
    lines.append("")
    lines.append("## Bytes to stopping (measured passes; mean [95% t-CI])")
    lines.append("")
    lines.append(
        "| dataset | arm | n | passes | cut 1 GB | cut 2 GB | cut 3 GB | total GB | ratio vs BP |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.dataset} | {r.arm} | {int(r.n)} | {r.passes_mean:.1f} | "
            f"{r.cut1_gb_mean:.1f} | {r.cut2_gb_mean:.1f} | {r.cut3_gb_mean:.1f} | "
            f"**{r.gb_mean:.1f}** [{r.gb_lo:.1f}, {r.gb_hi:.1f}] | "
            f"{r.ratio_vs_bp:.2f}× |"
        )
    lines.append("")
    lines.append(
        "`mf_seq` is the useful staged computation (M0 excluded); "
        "`mf_seq_as_implemented` adds the dead M0 forward. The core comparison is "
        "MF-Seq useful vs BP; the as-implemented row is the cost the current code "
        "actually pays."
    )
    lines.append("")
    lines.append("## MF-Seq stage contributions (as implemented)")
    lines.append("")
    lines.append("| dataset | stage | mean epochs | GB/pass | contribution GB | share |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in stages.iterrows():
        lines.append(
            f"| {r.dataset} | {r.stage} | {r.epochs_mean:.1f} | {r.gb_per_pass:.3f} | "
            f"{r.contribution_gb:.1f} | {r.share_pct:.1f}% |"
        )
    lines.append("")
    lines.append("## Cached-prefix counterfactual (augmentation disabled)")
    lines.append("")
    lines.append(
        "| dataset | fill per stage W2/W3 GB | cached read GB/epoch | recompute ex-M0 GB | cached ex-M0 GB | saving |"
    )
    lines.append("|---|---|---|---|---|---|")
    for _, r in cache_summary.iterrows():
        cmap = cache[cache.dataset == r.dataset].set_index("stage")
        lines.append(
            f"| {r.dataset} | {cmap.loc['W2','fill_gb']:.3f} / {cmap.loc['W3','fill_gb']:.3f} | "
            f"{cmap.loc['W2','cached_read_gb_per_epoch']:.3f} | "
            f"{r.recompute_ex_m0_gb:.1f} | {r.cached_ex_m0_gb:.1f} | "
            f"{r.ex_m0_saving_pct:.1f}% |"
        )
    lines.append("")
    lines.append(
        "Cached read is one a_{k-1} tensor per epoch (0.36 GB); fill is the prefix "
        "forward (0, 0.36, 0.72 GB for W1/W2/W3). CIFAR random crop/flip makes this "
        "invalid for canonical runs; it is a bound only."
    )
    lines.append("")
    lines.append("## Stage-0 forward waste and ex-M0 multipliers (ESTIMATES)")
    lines.append("")
    lines.append(
        "| dataset | M0 energy Wh | dead-forward FLOP share | dead-forward energy lower bound Wh | MF-Seq/BP energy | ex-M0 energy (est.) | MF-Seq/BP passes | ex-M0 passes (est.) | MF-Seq/BP time | ex-M0 time (est.) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in ex.iterrows():
        lines.append(
            f"| {r.dataset} | {r.m0_energy_wh:.2f} | {100*r.dead_forward_flop_share:.1f}% | "
            f"{r.dead_forward_energy_lower_bound_wh:.2f} | {r.energy_mult:.2f}× | "
            f"{r.ex_m0_energy_mult_estimate:.2f}× | {r.passes_mult:.2f}× | "
            f"{r.ex_m0_passes_mult_estimate:.2f}× | {r.time_mult:.2f}× | "
            f"{r.ex_m0_time_mult_estimate:.2f}× |"
        )
    lines.append("")
    lines.append(
        "The M0 code path runs a full no-grad forward through W1..W3 for every "
        "training and validation batch (`train_mf_matrix_only` calls "
        "`forward_with_intermediate_activations` and reads only a_0). The dead "
        "forward is 99.6% (C10) / 96.0% (C100) of M0-stage FLOPs; at the measured "
        "M0 energy this is a lower bound of 3.97 / 3.43 Wh of pure waste. The exact "
        "joule split is not observable without sub-stage NVML, so this is a "
        "FLOP-based bound, not a measured energy decomposition. Ex-M0 multipliers "
        "are **ESTIMATES**: they subtract the whole M0 stage under the current "
        "implementation and are not a production-pipeline measurement. Quote them "
        "with the estimate label, never as measured multipliers."
    )
    lines.append("")
    lines.append("## H-W3d pilot: cut bytes vs energy (sign only; no threshold)")
    lines.append("")
    lines.append(
        "| scope | cells | Spearman rho | descriptive p | bootstrap 95% CI | method | caveat |"
    )
    lines.append("|---|---:|---:|---:|---|---|---|")
    for _, r in h_w3d.iterrows():
        lines.append(
            f"| {r.scope} | {int(r.n_cells)} | {r.spearman_rho:+.3f} | "
            f"{r.p_descriptive:.3f} | [{r.rho_boot_lo:+.3f}, {r.rho_boot_hi:+.3f}] | "
            f"{r.bootstrap_method} | {r.caveat} |"
        )
    lines.append("")
    lines.append(
        "The pilot rho is negative: higher cut bytes do not track higher energy, "
        "because MF-Seq has the smallest useful cut bill but the largest measured "
        "energy. This is the expected direction for H-W3d, but the cell count is "
        "small and the two datasets share seeds. The seed bootstrap is degenerate "
        "because a common seed rescaling preserves the six-cell ordering; the "
        "cell bootstrap is shown only as a sensitivity. Report the sign and "
        "magnitude, not a one-sided test; no confirmatory p-value is attached."
    )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Cut communication is 2.16 GB/pass for BP (3 cuts × forward + backward), "
        "1.08 GB/pass for MF-Simul (3 forward activations, no backward across a "
        "cut), and 0.36 GB per prefix cut for each frozen-prefix stage epoch "
        "(W2 = 0.36, W3 = 0.72 GB/pass). "
        "MF-Seq's useful bytes-to-stopping are lower than BP's despite 4× the "
        "passes, because only the prefix tensors cross cuts and the backward pass "
        "never does. The dead M0 forward adds 1.08 GB/pass and, on the canonical "
        "runs, is the largest single cut-traffic term; removing it is a real "
        "engineering saving, not an accounting trick. The old 69 GB/pass figure "
        "must not be used as communication."
    )
    (args.out_dir / "w3_cut_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
