#!/usr/bin/env python3
"""Data Science Summit figures from G1 (2026-10) and PPAM-era (2026-09) data.

No W1 run, trace or contrast is read.  Inputs are restricted to the Pareto
tables produced by ``scripts/jpdc/dss_pareto.py`` and the G1 per-seed/W0b
tables.  The script refuses any input path containing ``w1`` or
``results/runs``.

The ex-M0 points are **estimates**, labelled as such on the figures:

* energy: measured MF-Seq energy scaled by ``1 - M0_share``, where ``M0_share``
  is the measured M0 share of the PPAM-era MF-Seq trace (W0b);
* accuracy: unchanged, because M0 is dead compute (no prediction path);
* passes: no ex-M0 estimate is drawn, because G1 did not instrument per-stage
  epoch counts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "jpdc_data"

ARMS = ["bp", "mf_joint", "mf_simul", "mf_seq"]
LABELS = {
    "bp": "BP",
    "mf_joint": "MF-Joint",
    "mf_simul": "MF-Simul",
    "mf_seq": "MF-Seq",
}
STYLE = {
    "bp": dict(marker="o", color="#1f77b4"),
    "mf_joint": dict(marker="s", color="#ff7f0e"),
    "mf_simul": dict(marker="^", color="#2ca02c"),
    "mf_seq": dict(marker="D", color="#d62728"),
}
DATASET_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
FOOTER = (
    "Sources: G1 2026-10 (Protocol A) and PPAM 2026-09 (Protocol B, M0 share). "
    "No W1 data."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard(paths: list[Path]) -> None:
    for path in paths:
        text = str(path.resolve())
        if "w1" in text.lower() or "results/runs" in text:
            raise SystemExit(f"refusing W1 input: {path}")


def _read_pareto(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    bad = set(frame["arm"]) - set(ARMS)
    if bad:
        raise SystemExit(f"{path}: unexpected arms {sorted(bad)}")
    numeric = [
        "test_accuracy_mean",
        "test_accuracy_lo",
        "test_accuracy_hi",
        "passes_mean",
        "passes_lo",
        "passes_hi",
        "energy_wh_mean",
        "energy_wh_lo",
        "energy_wh_hi",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _m0_share(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["experiment"].str.startswith("mf_cifar")].copy()
    frame["dataset"] = frame["experiment"].str.extract(
        r"mf_(cifar10|cifar100)_", expand=False
    )
    grouped = frame.groupby("dataset").agg(
        m0_wh=("M0_wh", "mean"),
        total_wh=("tidy_total_wh", "mean"),
        share=("M0_share", "mean"),
    )
    if set(grouped.index) != {"cifar10", "cifar100"}:
        raise SystemExit(f"{path}: expected cifar10 and cifar100 M0 rows")
    return grouped


def _row(frame: pd.DataFrame, dataset: str, arm: str) -> pd.Series | None:
    subset = frame[
        (frame["dataset"] == dataset)
        & (frame["arm"] == arm)
        & (frame["status"] == "measured")
    ]
    return None if subset.empty else subset.iloc[0]


def _errorbar(ax, row: pd.Series, x_col: str, label: str, **extra) -> None:
    x = row[f"{x_col}_mean"]
    xlo, xhi = row[f"{x_col}_lo"], row[f"{x_col}_hi"]
    y = row["test_accuracy_mean"]
    ylo, yhi = row["test_accuracy_lo"], row["test_accuracy_hi"]
    ax.errorbar(
        x,
        y,
        xerr=[[x - xlo], [xhi - x]],
        yerr=[[y - ylo], [yhi - y]],
        fmt=STYLE[row["arm"]]["marker"],
        color=STYLE[row["arm"]]["color"],
        ms=7,
        capsize=3,
        lw=1.0,
        label=label,
        **extra,
    )


def _scaled_errorbar(ax, row: pd.Series, factor: float, label: str) -> None:
    x = row["energy_wh_mean"] * factor
    xlo = row["energy_wh_lo"] * factor
    xhi = row["energy_wh_hi"] * factor
    y = row["test_accuracy_mean"]
    ax.errorbar(
        x,
        y,
        xerr=[[x - xlo], [xhi - x]],
        fmt="D",
        mfc="none",
        mec="#7f0000",
        ecolor="#7f0000",
        ms=9,
        capsize=3,
        lw=1.0,
        label=label,
    )
    # dotted connector from the measured MF-Seq point to the estimate
    ax.plot(
        [row["energy_wh_mean"], x],
        [y, y],
        ls=":",
        lw=1.0,
        color="0.45",
        zorder=1,
    )


def _finish(fig, out: Path) -> None:
    fig.text(0.01, 0.005, FOOTER, fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.85))
    for suffix in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


def figure_energy(protocol_a: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for ax, dataset in zip(axes, ("cifar10", "cifar100")):
        for arm in ARMS:
            row = _row(protocol_a, dataset, arm)
            if row is not None:
                _errorbar(ax, row, "energy_wh", LABELS[arm])
        ax.set_xlabel("Energy per run (Wh, NVML trace)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(DATASET_LABEL[dataset])
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        "Protocol A (G1, seeds 42–61): accuracy vs energy — measured arms only",
        fontsize=11,
        y=0.98,
    )
    fig.text(
        0.5,
        0.912,
        "No ex-M0 estimate in Protocol A (it would mix eras). The ex-M0 estimate is in the "
        "Protocol B figure.",
        ha="center",
        fontsize=7.8,
        color="0.35",
    )
    _finish(fig, out)


def figure_passes(protocol_a: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for ax, dataset in zip(axes, ("cifar10", "cifar100")):
        for arm in ARMS:
            row = _row(protocol_a, dataset, arm)
            if row is not None:
                _errorbar(ax, row, "passes", LABELS[arm])
        ax.set_xlabel("Data passes (epochs)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(DATASET_LABEL[dataset])
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        "Protocol A (G1): accuracy vs data passes — measured arms only",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.912,
        "No ex-M0 pass estimate is drawn: G1 did not instrument per-stage epoch counts. "
        "The ex-M0 energy estimate is in the Protocol B figure.",
        ha="center",
        fontsize=7.8,
        color="0.35",
    )
    _finish(fig, out)


def figure_protocol_b(protocol_b: pd.DataFrame, m0: pd.DataFrame, out: Path) -> None:
    present = [
        arm for arm in ARMS if _row(protocol_b, "cifar10", arm) is not None
        or _row(protocol_b, "cifar100", arm) is not None
    ]
    missing = [LABELS[arm] for arm in ARMS if arm not in present]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for ax, dataset in zip(axes, ("cifar10", "cifar100")):
        for arm in present:
            row = _row(protocol_b, dataset, arm)
            if row is not None:
                _errorbar(ax, row, "energy_wh", LABELS[arm])
        seq = _row(protocol_b, dataset, "mf_seq")
        if seq is not None:
            share = float(m0.loc[dataset, "share"])
            _scaled_errorbar(
                ax,
                seq,
                1.0 - share,
                f"MF-Seq ex-M0 — ESTIMATE ({100 * share:.1f}% removed)",
            )
        ax.set_xlabel("Energy per run (Wh, NVML trace)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(DATASET_LABEL[dataset])
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        "Protocol B (PPAM 2026-09, matched passes): accuracy vs energy",
        fontsize=11,
        y=0.98,
    )
    if missing:
        fig.text(
            0.5,
            0.912,
            "Not measured under Protocol B: "
            + ", ".join(missing)
            + ". The ex-M0 diamond is an estimate (measured M0 share, same era).",
            ha="center",
            fontsize=7.8,
            color="0.35",
        )
    _finish(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "analysis" / "dss_figures",
    )
    args = parser.parse_args()

    inputs = {
        "protocol_a": DATA / "dss" / "pareto_protocol_a.csv",
        "protocol_b": DATA / "dss" / "pareto_protocol_b.csv",
        "m0": DATA / "g1" / "w0b" / "w0b_stage_energy.csv",
    }
    _guard(list(inputs.values()))
    for path in inputs.values():
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    protocol_a = _read_pareto(inputs["protocol_a"])
    protocol_b = _read_pareto(inputs["protocol_b"])
    m0 = _m0_share(inputs["m0"])

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figure_energy(protocol_a, out / "dss_fig1_protocol_a_energy")
    figure_passes(protocol_a, out / "dss_fig2_protocol_a_passes")
    figure_protocol_b(protocol_b, m0, out / "dss_fig3_protocol_b_energy")

    readme = [
        "# DSS figure set — G1 + PPAM only",
        "",
        "**No W1 data.** These figures are built only from G1 (2026-10) and",
        "PPAM-era (2026-09) tables. The script refuses any path containing `w1`",
        "or `results/runs`.",
        "",
        "Every ex-M0 number is an **estimate**, marked on the figure:",
        "",
        "- energy (Protocol B only): measured MF-Seq energy × (1 − measured PPAM-era M0 share);",
        "- accuracy: assumed unchanged (M0 is dead compute);",
        "- passes: no ex-M0 estimate (G1 has no per-stage epoch counts).",
        "",
        "| figure | content |",
        "|---|---|",
        "| `dss_fig1_protocol_a_energy` | Protocol A (G1) accuracy vs energy, four measured arms (no ex-M0) |",
        "| `dss_fig2_protocol_a_passes` | Protocol A (G1) accuracy vs passes, measured arms only |",
        "| `dss_fig3_protocol_b_energy` | Protocol B (PPAM) accuracy vs energy, BP + MF-Seq + ex-M0 estimate |",
        "",
        "## M0 shares used",
        "",
        "| dataset | M0 Wh (mean) | trace Wh (mean) | M0 share |",
        "|---|---|---|---|",
    ]
    for dataset, row in m0.iterrows():
        readme.append(
            f"| {DATASET_LABEL[dataset]} | {row['m0_wh']:.3f} | "
            f"{row['total_wh']:.3f} | {row['share']:.4f} |"
        )
    readme.extend(
        [
            "",
            "## Inputs (SHA-256)",
            "",
            "| file | SHA-256 |",
            "|---|---|",
        ]
    )
    for path in inputs.values():
        readme.append(f"| `{path.name}` | `{_sha256(path)}` |")
    readme.extend(
        [
            "",
            "## Era caveat",
            "",
            "Protocol B is PPAM 2026-09 and its M0 share is from the same era, so",
            "the only ex-M0 estimate is drawn there. Protocol A (G1 2026-10) is",
            "shown as measured arms only, to avoid mixing eras.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(readme))
    print(f"wrote 3 figures (PDF+PNG) and README.md to {out}")
    for dataset, row in m0.iterrows():
        print(f"  {dataset}: M0 share {row['share']:.4f}")


if __name__ == "__main__":
    main()
