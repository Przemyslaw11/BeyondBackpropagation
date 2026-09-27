#!/usr/bin/env python3
"""The three manuscript figures, drawn from analysis/numbers.json.

Every value plotted here is read from numbers.json, which analyse.py derives
from artifacts/tidy/runs.csv. Nothing is typed in by hand.

Greyscale safety: every series is separated by hatch and by edge style as well
as by fill, so the figures survive a monochrome print.

Run:  python3 analysis/figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
NUM = json.load(open(HERE / "numbers.json"))

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

TEXTWIDTH = 4.8            # llncs \textwidth = 122 mm
CONFIGS = ["MNIST 2x1000", "Fashion-MNIST 2x1000", "CIFAR-10 3x2000", "CIFAR-100 3x2000"]
LABEL = {"MNIST 2x1000": "MNIST\n2$\\times$1000",
         "Fashion-MNIST 2x1000": "Fashion\n2$\\times$1000",
         "CIFAR-10 3x2000": "CIFAR-10\n3$\\times$2000",
         "CIFAR-100 3x2000": "CIFAR-100\n3$\\times$2000"}
RUNG_LABEL = ["BP", "BP-DS", "MF-Joint", "MF"]
RUNGS = ["bp", "bp_ds", "mf_joint", "mf_recompute"]

GREY = ["0.25", "0.55", "0.80"]
HATCH = ["", "///", "..."]


def saving(cg: str, metric: str) -> float:
    """Percentage by which MF is cheaper than BP at matched compute."""
    return -NUM["matched_compute"][cg]["metrics"][metric]["rel_pct"]


# --------------------------------------------------------------------------- #
def fig1() -> None:
    """Energy, memory and power saving against model scale."""
    metrics = [("total_gpu_energy_wh", "GPU energy"),
               ("peak_torch_alloc_mib", "Allocator memory"),
               ("trace_mean_power_w", "Mean board power")]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.35))
    x = np.arange(len(CONFIGS))
    width = 0.26
    for i, (m, lab) in enumerate(metrics):
        vals = [saving(cg, m) for cg in CONFIGS]
        ax.bar(x + (i - 1) * width, vals, width, label=lab,
               facecolor=GREY[i], edgecolor="black", linewidth=0.6, hatch=HATCH[i], zorder=3)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.annotate(f"{v:+.1f}".replace("-", "\u2212"), (xi, v), textcoords="offset points",
                        xytext=(0, 2 if v >= 0 else -10), ha="center", fontsize=7.5, zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABEL[c]}\n{NUM['params'][c]/1e6:.1f} M weights" for c in CONFIGS])
    ax.set_ylabel("MF saving over BP (%)")
    ax.set_ylim(-7, 38)
    ax.grid(axis="y", linestyle=":", color="0.7", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=1.6, columnspacing=1.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "CP025_fig1.pdf")
    plt.close(fig)
    print("[fig1] CP025_fig1.pdf")


# --------------------------------------------------------------------------- #
def fig2() -> None:
    """The BP -> MF ablation ladder: accuracy and energy, one change per step."""
    # Narrower than TEXTWIDTH: the tight bounding box grows past the text block
    # once the two y-labels and the legend below are added.
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2))
    markers = ["o", "s", "^", "D"]
    styles = ["-", "--", "-.", ":"]
    x = np.arange(len(RUNGS))

    ax = axes[0]
    for k, cg in enumerate(CONFIGS):
        base = NUM["ladder"][cg]["bp"]["test_accuracy"]["mean"]
        y = [NUM["ladder"][cg][r]["test_accuracy"]["mean"] - base for r in RUNGS]
        ax.plot(x, y, styles[k], marker=markers[k], color="black",
                markerfacecolor=["white", "0.6", "0.3", "black"][k],
                markeredgecolor="black", markeredgewidth=0.6, label=cg.split(" ")[0])
    ax.axhline(0, color="0.5", linewidth=0.6, linestyle="-")
    ax.set_xticks(x); ax.set_xticklabels(RUNG_LABEL, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy vs BP (pp)")
    ax.set_title("(a) Accuracy", loc="left")
    ax.grid(axis="y", linestyle=":", color="0.8")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    for k, cg in enumerate(CONFIGS):
        base = NUM["ladder"][cg]["bp"]["total_gpu_energy_wh"]["mean"]
        y = [100 * NUM["ladder"][cg][r]["total_gpu_energy_wh"]["mean"] / base for r in RUNGS]
        ax.plot(x, y, styles[k], marker=markers[k], color="black",
                markerfacecolor=["white", "0.6", "0.3", "black"][k],
                markeredgecolor="black", markeredgewidth=0.6)
    ax.axhline(100, color="0.5", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(RUNG_LABEL, rotation=20, ha="right")
    ax.set_ylabel("GPU energy (% of BP)")
    ax.set_title("(b) Energy", loc="left")
    ax.grid(axis="y", linestyle=":", color="0.8")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "CP025_fig2.pdf")
    plt.close(fig)
    print("[fig2] CP025_fig2.pdf")


# --------------------------------------------------------------------------- #
def fig3() -> None:
    """The same runs under both stopping protocols."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))
    x = np.arange(len(CONFIGS))
    width = 0.34

    ax = axes[0]
    for i, (key, lab, g, h) in enumerate(
            [("matched_compute", "Matched compute", "0.35", ""),
             ("own_convergence", "Own convergence", "0.85", "///")]):
        y = []
        for cg in CONFIGS:
            r = NUM[key][cg]["metrics"]["total_gpu_energy_wh"]
            y.append(100 * r["mf"]["mean"] / r["bp"]["mean"])
        ax.bar(x + (i - 0.5) * width, y, width, label=lab, facecolor=g,
               edgecolor="black", linewidth=0.6, hatch=h, zorder=3)
        for xi, v in zip(x + (i - 0.5) * width, y):
            ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6.2, zorder=4)
    ax.axhline(100, color="black", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CONFIGS], fontsize=6.5)
    ax.set_ylabel("MF GPU energy (% of BP)")
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 400])
    ax.set_yticklabels(["50", "100", "200", "400"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(40, 560)
    ax.set_title("(a) Energy", loc="left")
    ax.grid(axis="y", linestyle=":", color="0.8", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    for i, (key, lab, g, h) in enumerate(
            [("matched_compute", "Matched compute", "0.35", ""),
             ("own_convergence", "Own convergence", "0.85", "///")]):
        y = [NUM[key][cg]["metrics"]["test_accuracy"]["diff"] for cg in CONFIGS]
        lo = [NUM[key][cg]["metrics"]["test_accuracy"]["lo"] for cg in CONFIGS]
        hi = [NUM[key][cg]["metrics"]["test_accuracy"]["hi"] for cg in CONFIGS]
        err = np.array([np.array(y) - np.array(lo), np.array(hi) - np.array(y)])
        ax.bar(x + (i - 0.5) * width, y, width, yerr=err, capsize=1.6, label=lab,
               facecolor=g, edgecolor="black", linewidth=0.6, hatch=h, zorder=3,
               error_kw=dict(linewidth=0.6, capthick=0.6))
    ax.axhspan(-0.25, 0.25, color="0.88", zorder=1)
    ax.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CONFIGS], fontsize=6.5)
    ax.set_ylabel("Test accuracy vs BP (pp)")
    ax.set_title("(b) Accuracy", loc="left")
    ax.grid(axis="y", linestyle=":", color="0.8", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Patch(facecolor="0.35", edgecolor="black", linewidth=0.6, label="Matched compute"),
               Patch(facecolor="0.85", edgecolor="black", linewidth=0.6, hatch="///", label="Own convergence")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.12), handlelength=1.8)
    fig.tight_layout()
    fig.savefig(OUT / "CP025_fig3.pdf")
    plt.close(fig)
    print("[fig3] CP025_fig3.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
