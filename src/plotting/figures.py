"""Every camera-ready figure, generated from the tidy table and nothing else.

Reviewer R1 found Figures 3 and 4 illegible. The cause was not taste but
geometry: raster exports from the Weights & Biases UI, scaled down by
``\\includegraphics`` to a third of a 122 mm text block. Everything here is
vector, generated at its exact final width, and reads only
``artifacts/tidy/runs.csv``.

Three data hazards are enforced structurally rather than remembered:

* FF results produced before ``9a14332`` restored an epoch-2 network. Only
  ``results/phase4`` FF runs are eligible, and the W&B archive holds no FF
  per-epoch history newer than 2025-08, so FF convergence curves cannot be drawn
  at all -- :func:`ff_cost` shows the measured cost instead of inventing them.
* ``peak_gpu_mem_used_mib`` is a device-wide NVML reading dominated by the CUDA
  context. Memory claims use ``peak_torch_alloc_mib``; where the device-wide
  figure is plotted, the context floor is drawn so the reader can see it.
* "Epoch" means a different thing to each algorithm, because the cap is per
  stage. No x-axis here is an epoch count across algorithms.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from src.plotting import style, tidy
from src.plotting.tidy import RUNG_ORDER, RUNG_TRANSITIONS, Table

#: Pre-registered in scripts/analyze_ablation_ladder.py; tests assert they agree.
EQUIVALENCE_MARGIN_PP = 0.25
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260302

LADDER_CONFIGS: Tuple[str, ...] = (
    "MNIST 2x1000",
    "Fashion-MNIST 2x1000",
    "CIFAR-10 3x2000",
    "CIFAR-100 3x2000",
)

#: The two configurations that carry all six rungs. Random crop and horizontal
#: flip make a cached activation stale by construction, so the cache rungs are
#: structurally impossible on CIFAR rather than merely missing.
CACHEABLE_CONFIGS: Tuple[str, ...] = ("MNIST 2x1000", "Fashion-MNIST 2x1000")

#: RUNG_TRANSITIONS carries prose labels; at 122 mm they collide.
SHORT_TRANSITION: Tuple[str, ...] = (
    "aux.\nsuperv.",
    "readout\n$M_L$",
    "grad.\nlocality",
    "cache\n(dev.)",
    "cache\n(host)",
)

SHORT_CONFIG: Dict[str, str] = {
    "MNIST 2x1000": "MNIST\n2$\\times$1000",
    "Fashion-MNIST 2x1000": "Fashion\n2$\\times$1000",
    "CIFAR-10 3x2000": "CIFAR-10\n3$\\times$2000",
    "CIFAR-100 3x2000": "CIFAR-100\n3$\\times$2000",
}


# --- Statistics -------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(BOOTSTRAP_SEED)


def bootstrap_mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a mean."""
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return math.nan, math.nan, math.nan
    if sample.size == 1:
        return float(sample[0]), float(sample[0]), float(sample[0])
    draws = _rng().choice(sample, size=(N_BOOTSTRAP, sample.size), replace=True)
    means = draws.mean(axis=1)
    return float(sample.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def _paired(a: Dict[int, float], b: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    seeds = sorted(set(a) & set(b))
    return (
        np.array([a[s] for s in seeds], dtype=float),
        np.array([b[s] for s in seeds], dtype=float),
    )


def paired_difference_ci(
    a: Dict[int, float], b: Dict[int, float]
) -> Tuple[float, float, float, int]:
    """Bootstrap CI for mean(b) - mean(a), resampled by seed pair."""
    left, right = _paired(a, b)
    if left.size == 0:
        return math.nan, math.nan, math.nan, 0
    differences = right - left
    point, low, high = bootstrap_mean_ci(differences)
    return point, low, high, int(left.size)


def paired_ratio_ci(
    a: Dict[int, float], b: Dict[int, float]
) -> Tuple[float, float, float, int]:
    """Bootstrap CI for mean(b) / mean(a), resampled by seed pair."""
    left, right = _paired(a, b)
    if left.size == 0 or left.mean() == 0:
        return math.nan, math.nan, math.nan, 0
    index = _rng().integers(0, left.size, size=(N_BOOTSTRAP, left.size))
    ratios = right[index].mean(axis=1) / left[index].mean(axis=1)
    return (
        float(right.mean() / left.mean()),
        float(np.percentile(ratios, 2.5)),
        float(np.percentile(ratios, 97.5)),
        int(left.size),
    )


def equivalence_verdict(low: float, high: float, margin: float) -> str:
    """DIFFERENT, EQUIVALENT or INCONCLUSIVE. Never 'equal by non-significance'."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "inconclusive"
    if low > -margin and high < margin:
        return "equivalent"
    if low > 0 or high < 0:
        return "different"
    return "inconclusive"


# --- Table access -----------------------------------------------------------


def ladder(table: Table, config_group: str) -> Table:
    return table.where(phase="phase3_ladder", config_group=config_group)


def rung_values(table: Table, config_group: str, metric: str) -> Dict[str, Dict[int, float]]:
    """Per-rung {seed: value} for one ladder configuration."""
    scope = ladder(table, config_group)
    out: Dict[str, Dict[int, float]] = {}
    for rung in RUNG_ORDER:
        values = scope.where(rung=rung).values(metric)
        if values:
            out[rung] = values
    return out


def common_seeds(values: Dict[str, Dict[int, float]]) -> List[int]:
    """Seeds present in every rung, so one figure is internally paired."""
    if not values:
        return []
    shared = set.intersection(*(set(v) for v in values.values()))
    return sorted(shared)


def representative(table: Table, **filters) -> Optional[Dict[str, str]]:
    """The median-duration run of a group: a deterministic, unflattering pick."""
    scope = table.where(**filters)
    durations = scope.values("training_duration_sec")
    if not durations:
        return None
    ordered = sorted(durations.items(), key=lambda item: item[1])
    seed = ordered[len(ordered) // 2][0]
    rows = scope.where(seed=seed).rows
    return rows[0] if rows else None


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean; raw 5 Hz NVML traces are too noisy to read."""
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def trace_of(row: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Loads one run's NVML time series as finite numpy arrays."""
    raw = tidy.load_trace(row["monitoring_csv_path"])
    if not raw:
        return {}
    times = np.array(
        [t if t is not None else np.nan for t in raw["timestamp_sec"]], dtype=float
    )
    keep = np.isfinite(times)
    out = {"timestamp_sec": times[keep]}
    for name in raw:
        if name == "timestamp_sec":
            continue
        column = np.array(
            [v if v is not None else np.nan for v in raw[name]], dtype=float
        )
        out[name] = column[keep]
    return out


def _panel_label(axes, text: str) -> None:
    axes.text(
        -0.02,
        1.04,
        text,
        transform=axes.transAxes,
        fontsize=style.BASE_FONT_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _legend_entries(keys: Iterable[str]) -> Tuple[List[Line2D], List[str]]:
    handles, labels = [], []
    for key in keys:
        spec = style.series(key)
        handles.append(
            Line2D(
                [],
                [],
                color=spec.color,
                linestyle=spec.linestyle,
                marker=spec.marker,
                markersize=3,
            )
        )
        labels.append(spec.label)
    return handles, labels


# --- Figure 1: the ladder waterfall ----------------------------------------


def ladder_waterfall(table: Table, out_dir: Path) -> Path:
    """Where the BP -> MF energy change comes from, rung by rung.

    Phase 3 refuted the premise of the original task: under per-stage parity the
    total is an increase, not a saving, and locality accounts for essentially
    all of it. Both panels plot the increase.
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=style.figure_size(1.0, height_in=2.3)
    )

    headline = "CIFAR-10 3x2000"
    energies = rung_values(table, headline, "total_gpu_energy_wh")
    seeds = common_seeds(energies)
    rungs = [r for r in RUNG_ORDER if r in energies]
    means = [float(np.mean([energies[r][s] for s in seeds])) for r in rungs]
    baseline = means[0]
    relative = [100.0 * m / baseline for m in means]

    cursor = relative[0]
    left.bar(
        0,
        relative[0],
        width=0.62,
        color=style.series("bp").color,
        edgecolor="none",
        zorder=2,
    )
    left.text(
        0,
        relative[0] + 12,
        "100",
        ha="center",
        va="bottom",
        fontsize=style.SMALL_FONT_PT,
    )
    for index in range(1, len(rungs)):
        delta = relative[index] - cursor
        spec = style.series(rungs[index])
        left.plot(
            [index - 1 + 0.31, index + 0.31],
            [cursor, cursor],
            color="0.45",
            linewidth=0.5,
            zorder=1,
        )
        left.bar(
            index,
            delta,
            bottom=cursor,
            width=0.62,
            color=spec.color,
            edgecolor="none",
            zorder=2,
        )
        left.text(
            index,
            max(cursor, relative[index]) + 12,
            f"{delta:+.0f}",
            ha="center",
            va="bottom",
            fontsize=style.SMALL_FONT_PT,
        )
        cursor = relative[index]
    left.plot(
        [len(rungs) - 1 + 0.31, len(rungs) + 0.31],
        [cursor, cursor],
        color="0.45",
        linewidth=0.5,
        zorder=1,
    )
    left.bar(
        len(rungs),
        cursor,
        width=0.62,
        color="0.75",
        edgecolor="none",
        zorder=2,
    )
    left.text(
        len(rungs),
        cursor + 12,
        f"{cursor:.0f}",
        ha="center",
        va="bottom",
        fontsize=style.SMALL_FONT_PT,
    )

    left.set_xticks(range(len(rungs) + 1))
    left.set_xticklabels(
        ["BP"] + list(SHORT_TRANSITION[: len(rungs) - 1]) + ["MF\ntotal"],
        fontsize=style.SMALL_FONT_PT,
    )
    left.set_ylabel("Training energy\n(% of BP)")
    left.set_ylim(0, max(relative) * 1.22)
    left.axhline(100.0, color="0.4", linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
    _panel_label(left, "(a)")

    # Panel (b): the same decomposition for every configuration.
    positions = np.arange(len(RUNG_TRANSITIONS))
    width = 0.2
    for offset, config in enumerate(LADDER_CONFIGS):
        values = rung_values(table, config, "total_gpu_energy_wh")
        shared = common_seeds(values)
        contributions, xs = [], []
        for index, (source, target, _) in enumerate(RUNG_TRANSITIONS):
            if source not in values or target not in values:
                continue
            base = float(np.mean([values["bp"][s] for s in shared]))
            delta = float(
                np.mean([values[target][s] - values[source][s] for s in shared])
            )
            contributions.append(100.0 * delta / base)
            xs.append(positions[index] + (offset - 1.5) * width)
        right.bar(
            xs,
            contributions,
            width=width,
            label=config.replace("x", "$\\times$"),
            edgecolor="none",
            zorder=2,
        )
    right.axhline(0.0, color="0.3", linewidth=0.6, zorder=3)
    right.set_xticks(positions)
    right.set_xticklabels(SHORT_TRANSITION, fontsize=style.SMALL_FONT_PT)
    right.set_xlim(-0.75, len(RUNG_TRANSITIONS) - 0.25)
    right.set_ylabel("Contribution to energy\nchange (pp of BP)")
    right.legend(loc="upper left", fontsize=style.SMALL_FONT_PT, ncol=1)
    _panel_label(right, "(b)")

    return style.save(fig, out_dir / "fig_ladder_waterfall.pdf")


def ladder_power(table: Table, out_dir: Path) -> Path:
    """Why the totals invert: MF draws less power but runs for longer.

    Mean power and wall time are instrument-level quantities that survive any
    early-stopping choice. The ladder run summaries predate the Phase 4
    ``epochs_completed`` field, so Wh/epoch cannot be formed for these runs;
    mean power is its rate-equivalent and comes straight from NVML.
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=style.figure_size(1.0, height_in=2.05)
    )

    positions = np.arange(len(LADDER_CONFIGS))
    rungs_drawn: List[str] = []
    for metric, axes, label in (
        ("trace_mean_power_w", left, "Mean GPU power (W)"),
        ("training_duration_sec", right, "Training time (s)"),
    ):
        present = [
            rung
            for rung in RUNG_ORDER
            if any(rung in rung_values(table, c, metric) for c in LADDER_CONFIGS)
        ]
        rungs_drawn = present
        width = 0.8 / len(present)
        for order, rung in enumerate(present):
            centres, lows, highs, xs = [], [], [], []
            for index, config in enumerate(LADDER_CONFIGS):
                values = rung_values(table, config, metric)
                if rung not in values:
                    continue
                shared = common_seeds(
                    {k: v for k, v in values.items() if k in ("bp", rung)}
                )
                point, low, high = bootstrap_mean_ci(
                    [values[rung][s] for s in shared]
                )
                centres.append(point)
                lows.append(point - low)
                highs.append(high - point)
                xs.append(positions[index] + (order - (len(present) - 1) / 2) * width)
            spec = style.series(rung)
            axes.bar(
                xs,
                centres,
                width=width,
                color=spec.color,
                edgecolor="none",
                zorder=2,
            )
            axes.errorbar(
                xs,
                centres,
                yerr=[lows, highs],
                fmt="none",
                ecolor="0.25",
                elinewidth=0.5,
                capsize=1.2,
                zorder=3,
            )
        axes.set_xticks(positions)
        axes.set_xticklabels(
            [SHORT_CONFIG[c] for c in LADDER_CONFIGS], fontsize=style.SMALL_FONT_PT
        )
        axes.set_ylabel(label)

    left.set_ylim(0, None)
    right.set_yscale("log")
    _panel_label(left, "(a)")
    _panel_label(right, "(b)")

    handles, labels = _legend_entries(rungs_drawn)
    handles = [Patch(facecolor=style.series(r).color) for r in rungs_drawn]
    style.shared_legend(fig, handles, labels, ncol=len(rungs_drawn) // 2 or 1)
    fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    return style.save(fig, out_dir / "fig_ladder_power.pdf")


# --- Figure 2: the time-memory frontier -------------------------------------


def time_memory_frontier(table: Table, out_dir: Path) -> Path:
    """What gradient locality was supposed to buy, and what it actually buys.

    Joint gradients make a cached activation stale within one optimiser step, so
    rungs 1-3 have no cached variant at all; on CIFAR random crop and flip
    resample the input every epoch, so the cache rungs are impossible there for a
    second reason. That leaves MNIST and Fashion-MNIST as the only place the
    trade-off can be drawn -- and where it can be drawn, there is no frontier.
    Both cache rungs are slower *and* no cheaper in device memory than plain
    recomputation, which the shaded quadrant marks explicitly.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=style.figure_size(1.0, height_in=3.5), sharex="col"
    )
    configs = CACHEABLE_CONFIGS
    memory_metrics = (
        ("peak_torch_alloc_mib", "Peak device memory\n(MiB, torch allocator)"),
        ("trace_peak_process_rss_mib", "Peak host RSS\n(MiB, psutil)"),
    )

    drawn: List[str] = []
    for row_index, (metric, ylabel) in enumerate(memory_metrics):
        for column, config in enumerate(configs):
            panel = axes[row_index][column]
            times = rung_values(table, config, "training_duration_sec")
            memories = rung_values(table, config, metric)
            shared = common_seeds(times)
            means: Dict[str, Tuple[float, float]] = {}
            for rung in RUNG_ORDER:
                if rung not in times or rung not in memories:
                    continue
                if rung not in drawn:
                    drawn.append(rung)
                spec = style.series(rung)
                x_point, x_low, x_high = bootstrap_mean_ci(
                    [times[rung][s] for s in shared]
                )
                y_point, y_low, y_high = bootstrap_mean_ci(
                    [memories[rung][s] for s in shared if s in memories[rung]]
                )
                means[rung] = (x_point, y_point)
                panel.errorbar(
                    [x_point],
                    [y_point],
                    xerr=[[x_point - x_low], [x_high - x_point]],
                    yerr=[[y_point - y_low], [y_high - y_point]],
                    color=spec.color,
                    marker=spec.marker,
                    markersize=4.5,
                    markeredgecolor="white",
                    elinewidth=0.6,
                    capsize=1.5,
                    linestyle="none",
                    # BP and BP-DS land on top of each other; draw the ladder in
                    # reverse so the reference point stays visible.
                    zorder=3 + len(RUNG_ORDER) - RUNG_ORDER.index(rung),
                )
            panel.set_ylim(0, None)
            panel.set_xlim(0, None)
            # Everything up and to the right of plain recomputation is worse on
            # both axes at once. Caching lands there.
            if "mf_recompute" in means:
                x0, y0 = means["mf_recompute"]
                panel.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        panel.get_xlim()[1] - x0,
                        panel.get_ylim()[1] - y0,
                        facecolor="#CCCCCC",
                        alpha=0.5,
                        edgecolor="none",
                        zorder=0,
                    )
                )
                panel.text(
                    0.985,
                    0.955,
                    "dominated",
                    transform=panel.transAxes,
                    ha="right",
                    va="top",
                    fontsize=style.SMALL_FONT_PT,
                    color="0.35",
                )
            if column == 0:
                panel.set_ylabel(ylabel)
            if row_index == 1:
                panel.set_xlabel("Training time (s)")
    for panel, text in zip(
        (axes[0][0], axes[0][1], axes[1][0], axes[1][1]),
        (
            f"(a) {configs[0].replace('x', chr(215))}",
            f"(b) {configs[1].replace('x', chr(215))}",
            "(c)",
            "(d)",
        ),
    ):
        _panel_label(panel, text)

    handles = [
        Line2D(
            [],
            [],
            color=style.series(k).color,
            marker=style.series(k).marker,
            markersize=4,
            markeredgecolor="white",
            linestyle="none",
        )
        for k in drawn
    ]
    style.shared_legend(fig, handles, [style.series(k).label for k in drawn], ncol=3)
    fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
    return style.save(fig, out_dir / "fig_time_memory_frontier.pdf")


# --- Figure 3: the equivalence forest plot ----------------------------------


def _forest_rows_ladder(table: Table) -> List[Tuple[str, str, float, float, float, int]]:
    contrasts = (
        ("bp", "bp_ds", "BP-DS $-$ BP"),
        ("bp_ds", "mf_recompute", "MF $-$ BP-DS"),
        ("bp", "mf_recompute", "MF $-$ BP"),
        ("mf_recompute", "mf_cache_device", "cache-dev $-$ MF"),
        ("mf_recompute", "mf_cache_host", "cache-host $-$ MF"),
    )
    rows = []
    for source, target, label in contrasts:
        for config in LADDER_CONFIGS:
            values = rung_values(table, config, "test_accuracy")
            if source not in values or target not in values:
                continue
            point, low, high, n = paired_difference_ci(values[source], values[target])
            rows.append((label, SHORT_CONFIG[config].replace("\n", " "), point, low, high, n))
    return rows


def _forest_rows_phase4(table: Table) -> List[Tuple[str, str, float, float, float, int]]:
    pairs = (
        ("bp_mnist_mlp_3x1000", "ff_hinton_mnist_mlp_3x1000_adamw", "FF (AdamW)", "MNIST 3$\\times$1000"),
        ("bp_mnist_mlp_3x1000", "ff_hinton_mnist_mlp_3x1000_SGD", "FF (SGD)", "MNIST 3$\\times$1000"),
        ("bp_mnist_mlp_4x2000", "ff_hinton_mnist_mlp_4x2000", "FF", "MNIST 4$\\times$2000"),
        ("bp_fashion_mnist_mlp_4x2000", "ff_hinton_style_fashion_mnist_mlp_4x2000", "FF", "Fashion 4$\\times$2000"),
        ("bp_baseline_mnist_cnn_3block", "cafo_mnist_cnn_3block", "CaFo-Rand", "MNIST CNN"),
        ("bp_baseline_fashion_mnist_cnn_3block", "cafo_fashion_mnist_cnn_3block", "CaFo-Rand", "Fashion CNN"),
        ("bp_baseline_cifar10_cnn_3block", "cafo_cifar10_cnn_3block", "CaFo-Rand", "CIFAR-10 CNN"),
        ("bp_baseline_cifar100_cnn_3block", "cafo_cifar100_cnn_3block", "CaFo-Rand", "CIFAR-100 CNN"),
        ("bp_baseline_mnist_cnn_3block", "cafodfa_mnist_cnn_3block", "CaFo-DFA", "MNIST CNN"),
        ("bp_baseline_fashion_mnist_cnn_3block", "cafodfa_fashion_mnist_cnn_3block", "CaFo-DFA", "Fashion CNN"),
        ("bp_baseline_cifar10_cnn_3block", "cafodfa_cifar10_cnn_3block", "CaFo-DFA", "CIFAR-10 CNN"),
        ("bp_baseline_cifar100_cnn_3block", "cafodfa_cifar100_cnn_3block", "CaFo-DFA", "CIFAR-100 CNN"),
    )
    scope = table.where(phase="phase4")
    rows = []
    for baseline, method, label, config in pairs:
        left = scope.where(experiment_name=baseline).values("test_accuracy")
        right = scope.where(experiment_name=method).values("test_accuracy")
        if not left or not right:
            continue
        point, low, high, n = paired_difference_ci(left, right)
        rows.append((f"{label} $-$ BP", config, point, low, high, n))
    return rows


def _draw_forest(axes, entries, margin: float, xlabel: str) -> None:
    axes.axvspan(-margin, margin, color="#BBBBBB", alpha=0.45, zorder=0, linewidth=0)
    axes.axvline(0.0, color="0.25", linewidth=0.6, zorder=1)
    markers = {"equivalent": "o", "different": "s", "inconclusive": "D"}
    for index, (label, config, point, low, high, n) in enumerate(entries):
        y = len(entries) - index - 1
        verdict = equivalence_verdict(low, high, margin)
        colour = {
            "equivalent": style.OKABE_ITO["bluish_green"],
            "different": style.OKABE_ITO["vermillion"],
            "inconclusive": style.OKABE_ITO["blue"],
        }[verdict]
        axes.errorbar(
            [point],
            [y],
            xerr=[[point - low], [high - point]],
            color=colour,
            marker=markers[verdict],
            markersize=3.2,
            markerfacecolor=colour if verdict != "inconclusive" else "white",
            elinewidth=0.7,
            capsize=1.4,
            linestyle="none",
            zorder=3,
        )
    axes.set_yticks(range(len(entries)))
    axes.set_yticklabels(
        [f"{label}   {config}  ($n{{=}}{n}$)" for label, config, _, _, _, n in entries][
            ::-1
        ],
        fontsize=style.SMALL_FONT_PT,
    )
    axes.set_ylim(-0.7, len(entries) - 0.3)
    axes.set_xlabel(xlabel)
    axes.grid(axis="y", visible=False)


def equivalence_forest(table: Table, out_dir: Path) -> Path:
    """How a parity claim should be shown: an interval against a stated margin.

    A confidence interval that crosses zero is not evidence of equality. Hollow
    markers are the contrasts this seed count cannot resolve -- the interval
    covers both zero and a difference larger than the margin.
    """
    ladder_entries = _forest_rows_ladder(table)
    phase4_entries = _forest_rows_phase4(table)
    rows = len(ladder_entries) + len(phase4_entries)
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=style.figure_size(1.0, height_in=0.135 * rows + 1.15),
        height_ratios=[len(ladder_entries), len(phase4_entries)],
    )
    _draw_forest(top, ladder_entries, EQUIVALENCE_MARGIN_PP, "")
    _draw_forest(
        bottom, phase4_entries, EQUIVALENCE_MARGIN_PP, "Accuracy difference (pp)"
    )
    top.tick_params(labelbottom=True)
    _panel_label(top, "(a)")
    _panel_label(bottom, "(b)")

    handles = [
        Line2D([], [], color=style.OKABE_ITO["bluish_green"], marker="o", markersize=3.2, linestyle="none"),
        Line2D([], [], color=style.OKABE_ITO["vermillion"], marker="s", markersize=3.2, linestyle="none"),
        Line2D([], [], color=style.OKABE_ITO["blue"], marker="D", markersize=3.2, linestyle="none", markerfacecolor="white"),
        Patch(facecolor="#BBBBBB", alpha=0.45),
    ]
    labels = [
        "equivalent",
        "different",
        "inconclusive (underpowered)",
        f"$\\pm${EQUIVALENCE_MARGIN_PP} pp margin",
    ]
    style.shared_legend(fig, handles, labels, ncol=4)
    fig.get_layout_engine().set(rect=(0, 0.045 + 0.9 / rows, 1, 0.955 - 0.9 / rows))
    return style.save(fig, out_dir / "fig_equivalence_forest.pdf")


# --- Figure 4: FF ------------------------------------------------------------


#: A 5 Hz trace drawn with a dotted rung linestyle reads as a dot cloud. These
#: patterns stay separable in greyscale without dissolving the line.
TRACE_DASHES: Tuple = ("-", (0, (4.0, 1.6)), (0, (1.2, 1.3)), (0, (5.0, 1.2, 1.0, 1.2)))


def _plot_trace(
    axes,
    table: Table,
    entries: Sequence[Tuple[str, str]],
    column: str,
    smooth: int = 25,
) -> List[str]:
    """Draws one NVML column over wall-clock time for each named experiment."""
    drawn = []
    for experiment, series_key in entries:
        row = representative(table, experiment_name=experiment)
        if row is None:
            continue
        trace = trace_of(row)
        if not trace or column not in trace:
            continue
        values = trace[column]
        finite = np.isfinite(values)
        spec = style.series(series_key)
        axes.plot(
            trace["timestamp_sec"][finite],
            _rolling(values[finite], smooth),
            color=spec.color,
            linestyle=TRACE_DASHES[len(drawn) % len(TRACE_DASHES)],
            linewidth=1.0,
        )
        drawn.append(series_key)
    axes.set_xlabel("Wall-clock time (s)")
    axes.set_xlim(0, None)
    return drawn


def _trace_legend(keys: Sequence[str]) -> Tuple[List[Line2D], List[str]]:
    return (
        [
            Line2D(
                [],
                [],
                color=style.series(k).color,
                linestyle=TRACE_DASHES[i % len(TRACE_DASHES)],
            )
            for i, k in enumerate(keys)
        ],
        [style.series(k).label for k in keys],
    )


def ff_resource(table: Table, out_dir: Path) -> Path:
    """The replacement for the figure R1 named, and for the instrument it used.

    The submitted panel plotted the W&B agent's host-RSS stream in MB under an
    NVML mebibyte caption, which is why it showed BP near 815-925 while Table 1
    reported an NVML peak of 1168 MiB. Panel (a) puts the two instruments side by
    side: NVML sees the whole device, including a CUDA context neither algorithm
    controls, and only the torch allocator separates the two methods. Panel (b)
    is the throttling check the clock panel was there to make.
    """
    scope = table.where(phase="phase4")
    entries = (
        ("bp_fashion_mnist_mlp_4x2000", "bp"),
        ("ff_hinton_style_fashion_mnist_mlp_4x2000", "ff"),
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=style.figure_size(1.0, height_in=2.05)
    )

    instruments = (
        ("peak_gpu_mem_used_mib", "NVML,\ndevice-wide"),
        ("peak_torch_alloc_mib", "torch allocator,\nthis process"),
    )
    positions = np.arange(len(instruments))
    for order, (experiment, key) in enumerate(entries):
        centres, lows, highs = [], [], []
        for metric, _ in instruments:
            point, low, high = bootstrap_mean_ci(
                scope.where(experiment_name=experiment).series(metric)
            )
            centres.append(point)
            lows.append(point - low)
            highs.append(high - point)
        spec = style.series(key)
        xs = positions + (order - 0.5) * 0.36
        left.bar(xs, centres, width=0.36, color=spec.color, edgecolor="none", zorder=2)
        left.errorbar(
            xs,
            centres,
            yerr=[lows, highs],
            fmt="none",
            ecolor="0.25",
            elinewidth=0.5,
            capsize=1.2,
            zorder=3,
        )
        for x, value in zip(xs, centres):
            left.text(
                x,
                value + 18,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=style.SMALL_FONT_PT,
            )
    left.set_xticks(positions)
    left.set_xticklabels([label for _, label in instruments], fontsize=style.SMALL_FONT_PT)
    left.set_ylabel("Peak memory (MiB)")
    left.set_ylim(0, None)
    _panel_label(left, "(a)")

    drawn = _plot_trace(right, scope, entries, "sm_clock_mhz", smooth=51)
    right.set_ylabel("SM clock (MHz, NVML)")
    right.set_ylim(0, None)
    _panel_label(right, "(b)")

    handles, labels = _trace_legend([key for _, key in entries])
    style.shared_legend(fig, handles, labels, ncol=2)
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    return style.save(fig, out_dir / "fig_ff_resource.pdf")


def ff_cost(table: Table, out_dir: Path) -> Path:
    """FF's cost, measured. Its convergence curves no longer have a valid source.

    Every FF run with per-epoch history in the W&B archive predates ``9a14332``
    and therefore reports an epoch-2 network; the post-fix runs were archived
    without history. Rather than redraw an invalid curve, this shows what the
    valid runs actually cost.
    """
    scope = table.where(phase="phase4")
    pairs = (
        ("bp_mnist_mlp_3x1000", "ff_hinton_mnist_mlp_3x1000_adamw", "MNIST\n3$\\times$1000"),
        ("bp_mnist_mlp_4x2000", "ff_hinton_mnist_mlp_4x2000", "MNIST\n4$\\times$2000"),
        (
            "bp_fashion_mnist_mlp_4x2000",
            "ff_hinton_style_fashion_mnist_mlp_4x2000",
            "Fashion\n4$\\times$2000",
        ),
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=style.figure_size(1.0, height_in=2.0)
    )

    positions = np.arange(len(pairs))
    for order, (key, label) in enumerate((("bp", "BP"), ("ff", "FF"))):
        centres, lows, highs = [], [], []
        for baseline, method, _ in pairs:
            experiment = baseline if key == "bp" else method
            values = scope.where(experiment_name=experiment).series("test_accuracy")
            point, low, high = bootstrap_mean_ci(values)
            centres.append(point)
            lows.append(point - low)
            highs.append(high - point)
        spec = style.series(key)
        left.bar(
            positions + (order - 0.5) * 0.36,
            centres,
            width=0.36,
            color=spec.color,
            edgecolor="none",
            zorder=2,
        )
        left.errorbar(
            positions + (order - 0.5) * 0.36,
            centres,
            yerr=[lows, highs],
            fmt="none",
            ecolor="0.25",
            elinewidth=0.5,
            capsize=1.2,
            zorder=3,
        )
    left.set_xticks(positions)
    left.set_xticklabels([p[2] for p in pairs], fontsize=style.SMALL_FONT_PT)
    left.set_ylabel("Test accuracy (%)")
    left.set_ylim(0, 105)
    _panel_label(left, "(a)")

    for baseline, method, label in pairs[-1:]:
        for experiment, key in ((baseline, "bp"), (method, "ff")):
            row = representative(scope, experiment_name=experiment)
            if row is None:
                continue
            trace = trace_of(row)
            times = trace["timestamp_sec"]
            power = trace["power_watts"]
            finite = np.isfinite(power)
            energy = np.concatenate(
                [[0.0], np.cumsum(np.diff(times[finite]) * (
                    0.5 * (power[finite][1:] + power[finite][:-1])
                ))]
            ) / 3600.0
            spec = style.series(key)
            right.plot(
                times[finite],
                energy,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=1.0,
            )
    right.set_xlabel("Wall-clock time (s)")
    right.set_ylabel("Cumulative GPU energy\n(Wh, NVML)")
    right.set_xlim(0, None)
    right.set_ylim(0, None)
    _panel_label(right, "(b)")

    handles, labels = _legend_entries(("bp", "ff"))
    style.shared_legend(fig, handles, labels, ncol=2)
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    return style.save(fig, out_dir / "fig_ff_cost.pdf")


# --- Figure 5: CaFo ----------------------------------------------------------


def cafo_profile(table: Table, out_dir: Path) -> Path:
    """CaFo, rebuilt from the re-tuned Phase 4 runs.

    The back-ported thesis numbers are stale: CaFo is now 3.6-10.5x slower and
    2.9-10.9x more energy-hungry than BP on every dataset. Panel (b) shows the
    DFA block-pretraining stage as a distinct power regime -- the same claim the
    submitted caption made about a "flat initial segment", but measured.
    """
    scope = table.where(phase="phase4")
    datasets = (
        ("mnist", "MNIST"),
        ("fashion_mnist", "Fashion"),
        ("cifar10", "CIFAR-10"),
        ("cifar100", "CIFAR-100"),
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=style.figure_size(1.0, height_in=2.0),
        width_ratios=(1.3, 1.0),
    )

    positions = np.arange(len(datasets))
    variants = (
        ("bp_baseline_{}_cnn_3block", "bp"),
        ("cafo_{}_cnn_3block", "cafo_rand"),
        ("cafodfa_{}_cnn_3block", "cafo_dfa"),
    )
    for order, (template, key) in enumerate(variants):
        centres, lows, highs = [], [], []
        for slug, _ in datasets:
            values = scope.where(experiment_name=template.format(slug)).series(
                "test_accuracy"
            )
            point, low, high = bootstrap_mean_ci(values)
            centres.append(point)
            lows.append(point - low)
            highs.append(high - point)
        spec = style.series(key)
        xs = positions + (order - 1) * 0.27
        left.bar(xs, centres, width=0.27, color=spec.color, edgecolor="none", zorder=2)
        left.errorbar(
            xs,
            centres,
            yerr=[lows, highs],
            fmt="none",
            ecolor="0.25",
            elinewidth=0.5,
            capsize=1.2,
            zorder=3,
        )
    left.set_xticks(positions)
    left.set_xticklabels([d[1] for d in datasets], fontsize=style.SMALL_FONT_PT)
    left.set_ylabel("Test accuracy (%)")
    left.set_ylim(0, 105)
    left.set_xlim(-0.55, len(datasets) - 0.45)
    _panel_label(left, "(a)")

    entries = (
        ("bp_baseline_fashion_mnist_cnn_3block", "bp"),
        ("cafo_fashion_mnist_cnn_3block", "cafo_rand"),
        ("cafodfa_fashion_mnist_cnn_3block", "cafo_dfa"),
    )
    _plot_trace(right, scope, entries, "power_watts", smooth=25)
    right.set_ylabel("GPU power (W, NVML)")
    right.set_ylim(0, None)
    _panel_label(right, "(b)")

    handles, labels = _trace_legend([key for _, key in variants])
    style.shared_legend(fig, handles, labels, ncol=3)
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    return style.save(fig, out_dir / "fig_cafo_profile.pdf")


# --- Figure 6: MF hardware ---------------------------------------------------


def mf_hardware(table: Table, out_dir: Path) -> Path:
    """MF against BP at the instrument, on the CIFAR-10 3x2000 MLP.

    The temperature panel is gone: its 23-30 degC axis made a 3 degC difference
    look dramatic, and temperature is a consequence of power, not a cost. Power
    is the quantity the energy claim is actually made of.
    """
    scope = table.where(phase="phase3_ladder", config_group="CIFAR-10 3x2000")
    entries = (
        ("bp_cifar10_mlp_3x2000", "bp"),
        ("mf_cifar10_mlp_3x2000", "mf_recompute"),
    )
    fig, (power, util, memory) = plt.subplots(
        1, 3, figsize=style.figure_size(1.0, height_in=1.75)
    )

    drawn = _plot_trace(power, scope, entries, "power_watts")
    power.set_ylabel("GPU power (W)")
    _panel_label(power, "(a)")

    _plot_trace(util, scope, entries, "gpu_util_percent")
    util.set_ylabel("GPU utilisation (%)")
    _panel_label(util, "(b)")

    _plot_trace(memory, scope, entries, "gpu_mem_used_mib")
    memory.set_ylabel("GPU memory (MiB,\ndevice-wide incl. context)")
    _panel_label(memory, "(c)")

    for axes in (power, util, memory):
        # set_ylim(0, None) keeps matplotlib's margin on the *data* range, which
        # for a near-flat trace leaves the line sitting on the top spine.
        axes.set_ylim(0, axes.dataLim.y1 * 1.08)
        axes.xaxis.set_major_locator(plt.MaxNLocator(4))
        axes.yaxis.set_major_locator(plt.MaxNLocator(5))

    handles, labels = _trace_legend(drawn)
    style.shared_legend(fig, handles, labels, ncol=2)
    fig.get_layout_engine().set(rect=(0, 0.15, 1, 0.85))
    return style.save(fig, out_dir / "fig_mf_hardware.pdf")


# --- Diagnostics, for the response letter rather than the paper --------------


def diag_early_stopping(table: Table, out_dir: Path) -> Path:
    """How much of MF's cost is the algorithm and how much is the stopping rule.

    The harmonised protocol gives every algorithm the same per-stage patience,
    which lets MF run for ``hidden_layers + 1`` stages. The iso-compute protocol
    fixes the epoch budget instead. If MF's cost were an artefact of the stopping
    rule, these two bars would differ by roughly the cost ratio; they do not.
    """
    configs = (
        ("MNIST 2x1000", "mf_mnist_mlp_2x1000_equal_epochs"),
        ("Fashion-MNIST 2x1000", "mf_fashion_mnist_mlp_2x1000_equal_epochs"),
        ("CIFAR-10 3x2000", "mf_cifar10_mlp_3x2000_equal_epochs"),
        ("CIFAR-100 3x2000", "mf_cifar100_mlp_3x2000_equal_epochs"),
    )
    fig, axes = plt.subplots(1, 3, figsize=style.figure_size(1.0, height_in=2.0))
    metrics = (
        ("training_duration_sec", "Training time (s)"),
        ("total_gpu_energy_wh", "Total GPU energy (Wh)"),
        ("test_accuracy", "Test accuracy (%)"),
    )
    positions = np.arange(len(configs))
    series = (
        ("bp", "BP (harmonised)", style.OKABE_ITO["black"]),
        ("mf", "MF (harmonised)", style.OKABE_ITO["bluish_green"]),
        ("iso", "MF (iso-compute)", style.OKABE_ITO["reddish_purple"]),
    )
    for panel, (metric, label) in zip(axes, metrics):
        for order, (key, _, colour) in enumerate(series):
            centres, lows, highs = [], [], []
            for config, iso_experiment in configs:
                if key == "iso":
                    values = table.where(
                        phase="phase3_iso_compute", experiment_name=iso_experiment
                    ).series(metric)
                else:
                    rung = "bp" if key == "bp" else "mf_recompute"
                    values = list(
                        rung_values(table, config, metric).get(rung, {}).values()
                    )
                point, low, high = bootstrap_mean_ci(values)
                centres.append(point)
                lows.append(point - low)
                highs.append(high - point)
            xs = positions + (order - 1) * 0.27
            panel.bar(xs, centres, width=0.27, color=colour, edgecolor="none", zorder=2)
            panel.errorbar(
                xs,
                centres,
                yerr=[lows, highs],
                fmt="none",
                ecolor="0.25",
                elinewidth=0.5,
                capsize=1.2,
                zorder=3,
            )
        panel.set_xticks(positions)
        panel.set_xticklabels(
            [SHORT_CONFIG[c].replace("\n", " ") for c, _ in configs],
            fontsize=style.SMALL_FONT_PT,
            rotation=32,
            ha="right",
            rotation_mode="anchor",
        )
        panel.set_ylabel(label)
        panel.set_ylim(0, None)
    for panel, text in zip(axes, ("(a)", "(b)", "(c)")):
        _panel_label(panel, text)

    handles = [Patch(facecolor=colour) for _, _, colour in series]
    style.shared_legend(fig, handles, [name for _, name, _ in series], ncol=3)
    fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    return style.save(fig, out_dir / "fig_diag_early_stopping.pdf")


def diag_cache_strategy(table: Table, out_dir: Path) -> Path:
    """What caching the layer inputs actually buys, on the only two datasets
    where it is possible at all. Every bar is a paired change against plain
    recomputation, so a bar at zero means the strategy changed nothing.
    """
    metrics = (
        ("training_duration_sec", "time"),
        ("total_gpu_energy_wh", "energy"),
        ("peak_torch_alloc_mib", "device\nmem"),
        ("trace_peak_process_rss_mib", "host\nRSS"),
    )
    fig, axes = plt.subplots(1, 2, figsize=style.figure_size(1.0, height_in=2.0))
    positions = np.arange(len(metrics))
    strategies = ("mf_cache_device", "mf_cache_host")
    for panel, config in zip(axes, CACHEABLE_CONFIGS):
        for order, strategy in enumerate(strategies):
            centres, lows, highs = [], [], []
            for metric, _ in metrics:
                values = rung_values(table, config, metric)
                point, low, high, _ = paired_ratio_ci(
                    values["mf_recompute"], values[strategy]
                )
                centres.append(100.0 * (point - 1.0))
                lows.append(100.0 * (point - low))
                highs.append(100.0 * (high - point))
            spec = style.series(strategy)
            xs = positions + (order - 0.5) * 0.36
            panel.bar(
                xs, centres, width=0.36, color=spec.color, edgecolor="none", zorder=2
            )
            panel.errorbar(
                xs,
                centres,
                yerr=[lows, highs],
                fmt="none",
                ecolor="0.25",
                elinewidth=0.5,
                capsize=1.2,
                zorder=3,
            )
        panel.axhline(0.0, color="0.3", linewidth=0.6, zorder=3)
        panel.set_xticks(positions)
        panel.set_xticklabels(
            [label for _, label in metrics], fontsize=style.SMALL_FONT_PT
        )
        panel.set_ylabel("Change against MF\nrecomputation (%)")
    _panel_label(axes[0], f"(a) {CACHEABLE_CONFIGS[0].replace('x', chr(215))}")
    _panel_label(axes[1], f"(b) {CACHEABLE_CONFIGS[1].replace('x', chr(215))}")

    handles = [Patch(facecolor=style.series(s).color) for s in strategies]
    style.shared_legend(
        fig, handles, [style.series(s).label for s in strategies], ncol=2
    )
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    return style.save(fig, out_dir / "fig_diag_cache_strategy.pdf")


# --- Figure 7: the cost of training, against wall-clock time -----------------


def mf_bp_cost_curves(table: Table, out_dir: Path) -> Path:
    """What Figure 5 claimed to show, drawn from an instrument that can show it.

    The submitted convergence figure cannot be rebuilt: the Mono-Forward runs log
    ``Layer_M0/Val_LocalLoss_Epoch`` and nothing else per epoch. A layer-local
    objective is not the network's validation loss, and no global validation
    curve was ever recorded for MF, so no honest MF-versus-BP convergence plot
    exists. Cumulative NVML energy against wall-clock is the comparison that is
    instrumented identically for both, and it shows the same thing the totals do:
    MF's shallower slope never compensates for how much longer it runs.
    """
    scope = table.where(phase="phase3_ladder", config_group="CIFAR-10 3x2000")
    entries = (
        ("bp", "bp"),
        ("bp_ds", "bp_ds"),
        ("mf_joint", "mf_joint"),
        ("mf_recompute", "mf_recompute"),
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=style.figure_size(1.0, height_in=2.0)
    )

    drawn: List[str] = []
    for rung, key in entries:
        row = representative(scope, rung=rung)
        if row is None:
            continue
        trace = trace_of(row)
        if not trace:
            continue
        times, power = trace["timestamp_sec"], trace["power_watts"]
        finite = np.isfinite(power)
        times, power = times[finite], power[finite]
        energy = (
            np.concatenate(
                [[0.0], np.cumsum(np.diff(times) * 0.5 * (power[1:] + power[:-1]))]
            )
            / 3600.0
        )
        spec = style.series(key)
        left.plot(
            times,
            energy,
            color=spec.color,
            linestyle=TRACE_DASHES[len(drawn) % len(TRACE_DASHES)],
            linewidth=1.0,
        )
        left.plot(
            times[-1],
            energy[-1],
            color=spec.color,
            marker=spec.marker,
            markersize=3.5,
            markeredgecolor="white",
            linestyle="none",
        )
        drawn.append(key)
    left.set_xlabel("Wall-clock time (s)")
    left.set_ylabel("Cumulative GPU energy\n(Wh, NVML)")
    left.set_xlim(0, None)
    left.set_ylim(0, None)
    _panel_label(left, "(a)")

    # Panel (b): the same trade, aggregated over seeds rather than shown for one.
    for rung, key in entries:
        times = ladder(table, "CIFAR-10 3x2000").where(rung=rung).values(
            "training_duration_sec"
        )
        energies = ladder(table, "CIFAR-10 3x2000").where(rung=rung).values(
            "total_gpu_energy_wh"
        )
        shared = sorted(set(times) & set(energies))
        if not shared:
            continue
        spec = style.series(key)
        right.scatter(
            [times[s] for s in shared],
            [energies[s] for s in shared],
            color=spec.color,
            marker=spec.marker,
            s=9,
            linewidths=0,
            alpha=0.85,
            zorder=3,
        )
    right.set_xlabel("Training time (s)")
    right.set_ylabel("Total GPU energy\n(Wh, NVML)")
    right.set_xlim(0, None)
    right.set_ylim(0, None)
    # Iso-power guides, drawn once the scatter has settled the limits. They are
    # solid: any dash pattern here reads as a fourth algorithm.
    limit, top = right.get_xlim()[1], right.get_ylim()[1]
    for watts in (40, 60, 80):
        slope = watts / 3600.0
        x_end = min(limit, top / slope)
        right.plot([0, x_end], [0, slope * x_end], color="0.8", linewidth=0.6, zorder=1)
        # Labelled mid-span: both ends of panel (b) are occupied by the clusters.
        x_label = min(0.45 * limit, 0.9 * x_end)
        right.annotate(
            f"{watts} W",
            (x_label, slope * x_label),
            textcoords="offset points",
            xytext=(0, 2),
            fontsize=style.SMALL_FONT_PT,
            color="0.45",
            va="bottom",
            ha="center",
        )
    right.set_xlim(0, limit)
    right.set_ylim(0, top)
    _panel_label(right, "(b)")

    handles, labels = _trace_legend(drawn)
    style.shared_legend(fig, handles, labels, ncol=4)
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    return style.save(fig, out_dir / "fig_mf_bp_cost_curves.pdf")
