#!/usr/bin/env python3
"""Render the paper's result tables from the tidy table.

Figures already read from ``results/tidy/runs.csv``; the tables did not, so
every value in them was a hand transcription that no test could check. This
script closes that gap. The LaTeX it emits is committed and ``\\input`` by
main.tex, so a stale number shows up as a diff rather than as a claim.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from src.plotting.tidy import REPO_ROOT, Table

DEFAULT_TABLE = REPO_ROOT / "results" / "tidy" / "runs.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "tables"


@dataclass(frozen=True)
class Column:
    metric: str
    header: str
    decimals: int
    better: str


COLUMNS: Tuple[Column, ...] = (
    Column("test_accuracy", r"Accuracy (\%)", 2, "max"),
    Column("training_duration_sec", r"Time (\si{\second})", 1, "min"),
    Column("total_gpu_energy_wh", r"Energy (\si{\watt\hour})", 2, "min"),
    Column(
        "peak_torch_alloc_mib",
        r"\shortstack[c]{Peak Mem \\ (\si{\mebibyte})}",
        1,
        "min",
    ),
)

MEMORY_NOTE = (
    r"Peak Mem is \texttt{torch.cuda.max\_memory\_allocated}, the per-process "
    r"figure; allocation is deterministic, hence the zero SD."
)
ENERGY_NOTE = (
    r"Time and Energy cover full training to early stopping; energy is the NVML "
    r"power trace integrated over the run."
)


@dataclass(frozen=True)
class TableSpec:
    key: str
    label: str
    caption: str
    filters: Dict[str, str]
    arms: Tuple[Tuple[str, str], ...]
    extra: str = ""
    # "method" conflates the MF cache strategies, so ladder tables key on rung.
    arm_column: str = "method"
    # Heading for this block when the spec is rendered into the combined table.
    group: str = ""


SPECS: Tuple[TableSpec, ...] = (
    TableSpec(
        key="ff_bp_mlp_summary",
        label="tab:ff_bp_mlp_summary",
        caption=(
            r"Performance and efficiency: \FF{} vs.\ \BP{} on the Fashion-MNIST "
            r"4$\times$2000 MLP\@. "
        ),
        filters={"phase": "phase4", "config_group": "Fashion-MNIST 4x2000"},
        arms=(("ff", r"\FF{}-AdamW"), ("bp", r"\BP{} Baseline")),
        group=r"Fashion-MNIST 4$\times$2000 MLP",
    ),
    TableSpec(
        key="cafo_bp_cnn_summary",
        label="tab:cafo_bp_cnn_summary",
        caption=(
            r"Performance and efficiency: \CaFo{} variants vs.\ \BP{} on the "
            r"CIFAR-10 3-block CNN\@. "
        ),
        filters={"phase": "phase4", "config_group": "CIFAR-10 3-block CNN"},
        arms=(
            ("cafo_rand", r"\CaFoRand{}"),
            ("cafo_dfa", r"\CaFoDFA{}"),
            ("bp", r"\BP{} Baseline"),
        ),
        extra=(
            r"Time and Energy include the \CaFoDFA{} block-pretraining stage. "
            r"These values supersede the back-ported figures of the submitted "
            r"version, which were measured on an earlier cluster software stack."
        ),
        group=r"CIFAR-10 3-block CNN",
    ),
    TableSpec(
        key="mf_bp_perf_eff_summary",
        label="tab:mf_bp_perf_eff_summary",
        caption=(
            r"Performance and efficiency: \MF{} vs.\ \BP{} on the CIFAR-10 "
            r"3$\times$2000 MLP\@. "
        ),
        filters={"phase": "phase3_ladder", "config_group": "CIFAR-10 3x2000"},
        arms=(("mf_recompute", r"\MF{}"), ("bp", r"\BP{} Baseline")),
        arm_column="rung",
        extra=(
            r"CIFAR-10 inputs are flattened to 3072-dimensional vectors following "
            r"the \MF{} protocol. Under the harmonised stopping rule \MF{} is the "
            r"slower and more energy-hungry arm; Section~\ref{sec:ladder} "
            r"decomposes where that cost comes from."
        ),
        group=r"CIFAR-10 3$\times$2000 MLP",
    ),
)

COMBINED_KEY = "headline_summary"
COMBINED_LABEL = "tab:headline_summary"
COMBINED_CAPTION = (
    r"Each algorithm on its native architecture against an identically "
    r"configured, independently tuned \BP{} baseline, under the harmonised "
    r"stopping rule of Table~\ref{tab:early_stopping}. Mean~$\pm$~SD over the "
    r"stated seeds; the better result within each block is shown in "
    r"\textbf{bold}. CIFAR-10 inputs are flattened to 3072-dimensional vectors "
    r"for the MLP rows, following the \MF{} protocol, and \CaFo{} times include "
    r"the \CaFoDFA{} block-pretraining stage."
)


def _column_format(values: Sequence[float], decimals: int) -> str:
    digits = max(len(f"{value:.{decimals}f}".split(".")[0]) for value in values)
    return f"S[table-format={digits}.{decimals}]"


def _arm_statistics(
    scope: Table, column_name: str, arm: str
) -> Tuple[Dict[str, Tuple[float, float]], int]:
    summary: Dict[str, Tuple[float, float]] = {}
    counts = set()
    for column in COLUMNS:
        seeded = scope.where(**{column_name: arm}).values(column.metric)
        if not seeded:
            raise SystemExit(f"no {column.metric} for {column_name}={arm!r}")
        values = np.array([seeded[seed] for seed in sorted(seeded)], dtype=float)
        summary[column.metric] = (float(values.mean()), float(values.std(ddof=1)))
        counts.add(values.size)
    if len(counts) != 1:
        raise SystemExit(f"{arm}: metrics disagree on seed count {sorted(counts)}")
    return summary, counts.pop()


def render(spec: TableSpec, table: Table) -> str:
    scope = table.where(**spec.filters)
    statistics: Dict[str, Dict[str, Tuple[float, float]]] = {}
    seed_counts = set()
    for method, _ in spec.arms:
        summary, count = _arm_statistics(scope, spec.arm_column, method)
        statistics[method] = summary
        seed_counts.add(count)
    if len(seed_counts) != 1:
        raise SystemExit(f"{spec.key}: arms disagree on seed count {sorted(seed_counts)}")
    seeds = seed_counts.pop()

    best: Dict[str, str] = {}
    column_spec: list[str] = []
    for column in COLUMNS:
        means = {method: statistics[method][column.metric][0] for method, _ in spec.arms}
        chooser = max if column.better == "max" else min
        best[column.metric] = chooser(means, key=lambda m: means[m])
        column_spec.append(
            _column_format([m for m in means.values()], column.decimals)
            + r" @{\,$\pm$\,} "
            + _column_format(
                [statistics[m][column.metric][1] for m in means], column.decimals
            )
        )

    body = []
    for method, label in spec.arms:
        cells = [label]
        for column in COLUMNS:
            mean, deviation = statistics[method][column.metric]
            mark = r"\bfseries " if best[column.metric] == method else ""
            cells.append(f"{mark}{mean:.{column.decimals}f}")
            cells.append(f"{mark}{deviation:.{column.decimals}f}")
        body.append("    " + " & ".join(cells) + r" \\")

    headers = " &\n      ".join(
        rf"\multicolumn{{2}}{{c}}{{{column.header}}}" for column in COLUMNS
    )
    caption = (
        f"{spec.caption}"
        rf"Mean~$\pm$~SD over \num{{{seeds}}} seeds. "
        r"Better result in each column is shown in \textbf{bold}. "
        f"{ENERGY_NOTE} {MEMORY_NOTE}"
        + (f" {spec.extra}" if spec.extra else "")
    )
    return "\n".join(
        [
            "% Generated by scripts/make_tables.py from results/tidy/runs.csv.",
            "% Do not edit by hand; edit the generator and re-run it.",
            r"\begin{table}[tb]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            rf"  \label{{{spec.label}}}",
            r"  \setlength{\tabcolsep}{4pt}%",
            r"  \small",
            r"  \begin{tabular}{l",
            "      " + "\n      ".join(column_spec) + "}",
            r"    \toprule",
            "    {Algorithm} &\n      " + headers + r" \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_combined(specs: Sequence[TableSpec], table: Table) -> str:
    """Render every spec as one table, so the paper pays for one float."""
    blocks: list[Tuple[TableSpec, Dict[str, Dict[str, Tuple[float, float]]], int]] = []
    for spec in specs:
        scope = table.where(**spec.filters)
        statistics: Dict[str, Dict[str, Tuple[float, float]]] = {}
        seed_counts = set()
        for method, _ in spec.arms:
            summary, count = _arm_statistics(scope, spec.arm_column, method)
            statistics[method] = summary
            seed_counts.add(count)
        if len(seed_counts) != 1:
            raise SystemExit(f"{spec.key}: arms disagree on seed count {sorted(seed_counts)}")
        blocks.append((spec, statistics, seed_counts.pop()))

    column_spec: list[str] = []
    for column in COLUMNS:
        means = [
            statistics[method][column.metric][0]
            for spec, statistics, _ in blocks
            for method, _ in spec.arms
        ]
        deviations = [
            statistics[method][column.metric][1]
            for spec, statistics, _ in blocks
            for method, _ in spec.arms
        ]
        column_spec.append(
            _column_format(means, column.decimals)
            + r" @{\,$\pm$\,} "
            + _column_format(deviations, column.decimals)
        )

    span = 1 + 2 * len(COLUMNS)
    body: list[str] = []
    for index, (spec, statistics, seeds) in enumerate(blocks):
        if index:
            body.append(r"    \addlinespace")
        body.append(
            rf"    \multicolumn{{{span}}}{{l}}{{\itshape {spec.group} "
            rf"($n=\num{{{seeds}}}$)}} \\"
        )
        best: Dict[str, str] = {}
        for column in COLUMNS:
            means = {m: statistics[m][column.metric][0] for m, _ in spec.arms}
            chooser = max if column.better == "max" else min
            best[column.metric] = chooser(means, key=lambda m: means[m])
        for method, label in spec.arms:
            cells = [label]
            for column in COLUMNS:
                mean, deviation = statistics[method][column.metric]
                mark = r"\bfseries " if best[column.metric] == method else ""
                cells.append(f"{mark}{mean:.{column.decimals}f}")
                cells.append(f"{mark}{deviation:.{column.decimals}f}")
            body.append("    " + " & ".join(cells) + r" \\")

    headers = " &\n      ".join(
        rf"\multicolumn{{2}}{{c}}{{{column.header}}}" for column in COLUMNS
    )
    caption = f"{COMBINED_CAPTION} {ENERGY_NOTE} {MEMORY_NOTE}"
    return "\n".join(
        [
            "% Generated by scripts/make_tables.py from results/tidy/runs.csv.",
            "% Do not edit by hand; edit the generator and re-run it.",
            r"\begin{table}[tb]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            rf"  \label{{{COMBINED_LABEL}}}",
            r"  \setlength{\tabcolsep}{4pt}%",
            r"  \small",
            r"  \begin{tabular}{l",
            "      " + "\n      ".join(column_spec) + "}",
            r"    \toprule",
            "    {Algorithm} &\n      " + headers + r" \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()

    table = Table.load(arguments.table).where(superseded="")
    arguments.out.mkdir(parents=True, exist_ok=True)
    rendered = {spec.key: render(spec, table) for spec in SPECS}
    rendered[COMBINED_KEY] = render_combined(SPECS, table)
    for key, text in rendered.items():
        path = arguments.out / f"{key}.tex"
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        print(f"{path.relative_to(REPO_ROOT)}  {len(text):5d} B  {digest}")


if __name__ == "__main__":
    main()
