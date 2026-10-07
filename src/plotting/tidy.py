"""The aggregator: one tidy long-format table behind every figure and table.

The training engine emits a per-run summary JSON and an NVML time
series; the study phases filled ``results/`` with 916 of them. This module is the
join. It carries, for every value it emits, the file it came from, the
instrument that produced it and the unit it is in, so a figure can never
silently mix a device-wide NVML reading with a host RSS one -- the exact defect
that put a host-memory trace under an "NVML peak memory" caption in the
submitted paper.

Long format, one row per (run, metric). Identity columns let known non-canonical
protocol runs be filtered rather than silently absorbed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Provenance -------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """What a metric means, in what unit, measured by what."""

    key: str
    unit: str
    instrument: str
    note: str = ""


#: Metrics read straight out of a run-summary JSON.
SUMMARY_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("test_accuracy", "percent", "held-out test split", "engine"),
    MetricSpec("test_loss", "nats", "held-out test split", "nan for pre-4373312 MF"),
    MetricSpec("training_duration_sec", "s", "host monotonic clock", "training only"),
    MetricSpec("total_run_duration_sec", "s", "host monotonic clock", "incl. setup"),
    MetricSpec(
        "total_gpu_energy_wh",
        "Wh",
        "NVML nvmlDeviceGetPowerUsage",
        "trapezoidal integral of the 5 Hz power trace",
    ),
    MetricSpec(
        "total_gpu_energy_joules", "J", "NVML nvmlDeviceGetPowerUsage", "as above"
    ),
    MetricSpec(
        "peak_gpu_mem_used_mib",
        "MiB",
        "NVML nvmlDeviceGetMemoryInfo",
        "DEVICE-WIDE: includes the ~800 MiB CUDA context, not an algorithm's footprint",
    ),
    MetricSpec(
        "peak_torch_alloc_mib",
        "MiB",
        "torch.cuda.max_memory_allocated",
        "process-local allocator high-water mark; the honest memory figure",
    ),
    MetricSpec(
        "peak_process_rss_mib",
        "MiB",
        "psutil Process.memory_info().rss",
        "HOST memory, not GPU memory",
    ),
    MetricSpec(
        "peak_gpu_util_percent", "percent", "NVML nvmlDeviceGetUtilizationRates", ""
    ),
    MetricSpec(
        "codecarbon_emissions_gCO2e", "gCO2e", "CodeCarbon offline tracker", "POL grid"
    ),
    MetricSpec(
        "epochs_completed",
        "epoch",
        "engine counter",
        "SUMMED OVER STAGES for MF/CaFo/FF; not comparable across algorithms",
    ),
    MetricSpec(
        "gpu_energy_wh_per_epoch",
        "Wh/epoch",
        "derived: total_gpu_energy_wh / epochs_completed",
        "cost per data pass; protocol-independent",
    ),
    MetricSpec(
        "training_sec_per_epoch",
        "s/epoch",
        "derived: training_duration_sec / epochs_completed",
        "cost per data pass; protocol-independent",
    ),
)

#: Metrics derived by reducing a run's NVML time series.
TRACE_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        "trace_mean_power_w",
        "W",
        "NVML nvmlDeviceGetPowerUsage",
        "trapezoidal energy / trace duration",
    ),
    MetricSpec("trace_duration_sec", "s", "host monotonic clock", "sampling window"),
    MetricSpec("trace_samples", "count", "monitor thread", ""),
    MetricSpec(
        "trace_peak_gpu_mem_used_mib",
        "MiB",
        "NVML nvmlDeviceGetMemoryInfo",
        "DEVICE-WIDE, includes the CUDA context",
    ),
    MetricSpec(
        "trace_mean_gpu_util_percent",
        "percent",
        "NVML nvmlDeviceGetUtilizationRates",
        "",
    ),
    MetricSpec("trace_mean_sm_clock_mhz", "MHz", "NVML nvmlDeviceGetClockInfo", "SM"),
    MetricSpec(
        "trace_sd_sm_clock_mhz",
        "MHz",
        "NVML nvmlDeviceGetClockInfo",
        "clock volatility",
    ),
    MetricSpec(
        "trace_mean_gpu_temp_celsius", "degC", "NVML nvmlDeviceGetTemperature", ""
    ),
    MetricSpec(
        "trace_peak_process_rss_mib",
        "MiB",
        "psutil Process.memory_info().rss",
        "HOST memory",
    ),
)

METRIC_SPECS: Dict[str, MetricSpec] = {
    spec.key: spec for spec in SUMMARY_METRICS + TRACE_METRICS
}

#: Columns of the NVML per-run CSV, and what produced each one.
TRACE_COLUMNS: Dict[str, Tuple[str, str]] = {
    "timestamp_sec": ("s", "host monotonic clock"),
    "power_watts": ("W", "NVML nvmlDeviceGetPowerUsage"),
    "gpu_util_percent": ("percent", "NVML nvmlDeviceGetUtilizationRates"),
    "mem_util_percent": ("percent", "NVML nvmlDeviceGetUtilizationRates"),
    "gpu_mem_used_mib": ("MiB", "NVML nvmlDeviceGetMemoryInfo (device-wide)"),
    "gpu_temp_celsius": ("degC", "NVML nvmlDeviceGetTemperature"),
    "sm_clock_mhz": ("MHz", "NVML nvmlDeviceGetClockInfo"),
    "compute_processes": ("count", "NVML nvmlDeviceGetComputeRunningProcesses"),
    "process_rss_mib": ("MiB", "psutil Process.memory_info().rss (HOST)"),
}


# --- Sources ----------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A results directory and the experimental protocol it was produced under."""

    root: str
    phase: str
    protocol: str
    superseded: str = ""

    @property
    def name(self) -> str:
        return self.root.rsplit("/", 1)[-1]


SOURCES: Tuple[Source, ...] = (
    Source("results/runs", "phase3_ladder", "harmonised"),
    Source("results/phase4", "phase4", "harmonised"),
    Source("results/reproduction", "phase4_reproduction", "legacy"),
    Source(
        "results/reproduction_m0mem_bug",
        "phase4_reproduction",
        "legacy",
        superseded="M0-only memory sampling, fixed in 6f0a660",
    ),
    Source("results/equal_epochs", "phase3_iso_compute", "iso_compute"),
    Source(
        "results/equal_epochs_noval",
        "phase3_iso_compute",
        "iso_compute_noval",
        superseded="validation pass disabled with early stopping; confounded",
    ),
)

#: Ladder rung identity, keyed on (algorithm, activation_cache).
#: Mirrors ``_rung_key`` in scripts/analyze_ablation_ladder.py; the pre-registered
#: script owns the definition and tests/test_plotting.py asserts they agree.
RUNG_ORDER: Tuple[str, ...] = (
    "bp",
    "bp_ds",
    "mf_joint",
    "mf_recompute",
    "mf_cache_device",
    "mf_cache_host",
)

#: What each adjacent rung transition isolates.
RUNG_TRANSITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("bp", "bp_ds", "auxiliary\nsupervision"),
    ("bp_ds", "mf_joint", "readout\n($M_L$)"),
    ("mf_joint", "mf_recompute", "gradient\nlocality"),
    ("mf_recompute", "mf_cache_device", "cache\n(device)"),
    ("mf_cache_device", "mf_cache_host", "cache\n(host)"),
)

DATASET_LABELS: Dict[str, str] = {
    "mnist": "MNIST",
    "fashionmnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}


def rung_of(algorithm: str, activation_cache: str) -> str:
    """Maps an (algorithm, cache) pair onto its ladder rung, or '' if off-ladder."""
    algorithm = (algorithm or "").lower()
    cache = (activation_cache or "recompute").lower()
    if algorithm in ("bp", "bp_ds", "mf_joint"):
        return algorithm
    if algorithm == "mf":
        return {
            "recompute": "mf_recompute",
            "cache_device": "mf_cache_device",
            "cache_host": "mf_cache_host",
        }.get(cache, "")
    return ""


def architecture_label(architecture: Sequence[int], experiment_name: str) -> str:
    """A compact, human-readable architecture name."""
    dims = list(architecture or [])
    if not dims:
        return "unknown"
    if "cnn" in experiment_name.lower():
        return f"{len(dims)}-block CNN"
    if len(set(dims)) == 1:
        return f"{len(dims)}x{dims[0]}"
    return "-".join(str(d) for d in dims)


def method_of(experiment_name: str, algorithm: str) -> str:
    """The plotted series key: CaFo and FF variants are distinct methods."""
    name = experiment_name.lower()
    algorithm = (algorithm or "").lower()
    if algorithm == "cafo":
        return "cafo_dfa" if "dfa" in name else "cafo_rand"
    if algorithm == "ff":
        return "ff"
    return rung_of(algorithm, "recompute") or algorithm


# --- Records ----------------------------------------------------------------


@dataclass
class Row:
    """One (run, metric) observation, with its provenance attached."""

    run_id: str
    source_file: str
    phase: str
    protocol: str
    superseded: str
    experiment_name: str
    algorithm: str
    method: str
    rung: str
    cache_strategy: str
    dataset: str
    architecture: str
    config_group: str
    seed: int
    monitoring_csv_path: str
    metric: str
    value: float
    unit: str
    instrument: str
    note: str


COLUMNS: Tuple[str, ...] = tuple(f.name for f in fields(Row))


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resolve(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.is_file() else None


def load_trace(path_text: str) -> Dict[str, List[float]]:
    """Reads one NVML per-run CSV into column arrays, dropping blank cells."""
    path = _resolve(path_text)
    if path is None:
        return {}
    columns: Dict[str, List[float]] = {name: [] for name in TRACE_COLUMNS}
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            for name in columns:
                columns[name].append(_finite(record.get(name)))
    return columns


def _trapezoid(times: Sequence[Optional[float]], values: Sequence[Optional[float]]):
    """Integral and covered duration, skipping segments with missing data."""
    total = 0.0
    covered = 0.0
    for index in range(len(times) - 1):
        t0, t1 = times[index], times[index + 1]
        v0, v1 = values[index], values[index + 1]
        if t0 is None or t1 is None or t1 <= t0 or v0 is None or v1 is None:
            continue
        total += 0.5 * (v0 + v1) * (t1 - t0)
        covered += t1 - t0
    return total, covered


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _sd(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [v for v in values if v is not None]
    if len(kept) < 2:
        return None
    mean = sum(kept) / len(kept)
    return math.sqrt(sum((v - mean) ** 2 for v in kept) / (len(kept) - 1))


def _peak(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [v for v in values if v is not None]
    return max(kept) if kept else None


def trace_metrics(trace: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    """Reduces one NVML time series to the scalars the figures need."""
    if not trace or len(trace.get("timestamp_sec", [])) < 2:
        return {}
    times = trace["timestamp_sec"]
    energy_j, covered = _trapezoid(times, trace["power_watts"])
    span = _finite(times[-1]) if times[-1] is not None else None
    return {
        "trace_mean_power_w": energy_j / covered if covered > 0 else None,
        "trace_duration_sec": span,
        "trace_samples": float(len(times)),
        "trace_peak_gpu_mem_used_mib": _peak(trace["gpu_mem_used_mib"]),
        "trace_mean_gpu_util_percent": _mean(trace["gpu_util_percent"]),
        "trace_mean_sm_clock_mhz": _mean(trace["sm_clock_mhz"]),
        "trace_sd_sm_clock_mhz": _sd(trace["sm_clock_mhz"]),
        "trace_mean_gpu_temp_celsius": _mean(trace["gpu_temp_celsius"]),
        "trace_peak_process_rss_mib": _peak(trace["process_rss_mib"]),
    }


def _iter_summaries(source: Source) -> Iterator[Tuple[Path, Dict]]:
    root = REPO_ROOT / source.root
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("error"):
            continue
        yield path, record


def build(sources: Sequence[Source] = SOURCES, with_traces: bool = True) -> List[Row]:
    """Builds the tidy table. One row per (run, metric)."""
    rows: List[Row] = []
    for source in sources:
        for path, record in _iter_summaries(source):
            relative = path.relative_to(REPO_ROOT).as_posix()
            experiment = str(record.get("experiment_name", path.stem))
            algorithm = str(record.get("algorithm", ""))
            cache = str(record.get("activation_cache", "recompute"))
            dataset = str(record.get("dataset", "unknown"))
            architecture = architecture_label(record.get("architecture"), experiment)
            seed = int(record.get("seed", -1))
            csv_path = str(record.get("monitoring_csv_path", ""))
            identity = {
                "run_id": f"{source.name}/{experiment}/seed{seed}",
                "source_file": relative,
                "phase": source.phase,
                "protocol": source.protocol,
                "superseded": source.superseded,
                "experiment_name": experiment,
                "algorithm": algorithm,
                "method": method_of(experiment, algorithm),
                "rung": rung_of(algorithm, cache),
                "cache_strategy": cache,
                "dataset": DATASET_LABELS.get(dataset.lower(), dataset),
                "architecture": architecture,
                "config_group": f"{DATASET_LABELS.get(dataset.lower(), dataset)} "
                f"{architecture}",
                "seed": seed,
                "monitoring_csv_path": csv_path,
            }

            values: Dict[str, Optional[float]] = {
                spec.key: _finite(record.get(spec.key)) for spec in SUMMARY_METRICS
            }
            if with_traces:
                values.update(trace_metrics(load_trace(csv_path)))

            for key, value in values.items():
                if value is None:
                    continue
                spec = METRIC_SPECS[key]
                rows.append(
                    Row(
                        metric=key,
                        value=value,
                        unit=spec.unit,
                        instrument=spec.instrument,
                        note=spec.note,
                        **identity,
                    )
                )
    rows.sort(key=lambda row: (row.run_id, row.metric))
    return rows


def write(rows: Sequence[Row], path: Path) -> Path:
    """Writes the tidy table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in COLUMNS})
    return path


def write_provenance(path: Path) -> Path:
    """Writes the metric dictionary: unit and instrument for every column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "unit", "instrument", "note", "origin"])
        for spec in SUMMARY_METRICS:
            writer.writerow(
                [spec.key, spec.unit, spec.instrument, spec.note, "run summary JSON"]
            )
        for spec in TRACE_METRICS:
            writer.writerow(
                [spec.key, spec.unit, spec.instrument, spec.note, "NVML per-run CSV"]
            )
        for name, (unit, instrument) in TRACE_COLUMNS.items():
            writer.writerow(
                [f"csv:{name}", unit, instrument, "", "NVML per-run CSV column"]
            )
    return path


# --- Reading back -----------------------------------------------------------


def read(path: Path) -> List[Dict[str, str]]:
    """Reads the tidy table back. Figures consume this, never the raw JSONs."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Table:
    """A queryable view over the tidy table."""

    def __init__(self, rows: Sequence[Dict[str, str]]):
        self.rows = list(rows)

    @classmethod
    def load(cls, path: Path) -> "Table":
        return cls(read(path))

    def where(self, **filters) -> "Table":
        """Keeps rows matching every filter. A tuple/list value means 'in'."""

        def keep(row: Dict[str, str]) -> bool:
            for key, wanted in filters.items():
                actual = row.get(key, "")
                if isinstance(wanted, (tuple, list, set, frozenset)):
                    if actual not in {str(w) for w in wanted}:
                        return False
                elif actual != str(wanted):
                    return False
            return True

        return Table([row for row in self.rows if keep(row)])

    def values(self, metric: str) -> Dict[int, float]:
        """Metric values keyed by seed. Duplicate seeds would be a join bug."""
        out: Dict[int, float] = {}
        for row in self.rows:
            if row["metric"] != metric:
                continue
            seed = int(row["seed"])
            if seed in out:
                raise ValueError(
                    f"duplicate seed {seed} for {metric} in {row['run_id']}; "
                    "the filter is not narrow enough to identify a single run"
                )
            out[seed] = float(row["value"])
        return out

    def series(self, metric: str) -> List[float]:
        return [value for _, value in sorted(self.values(metric).items())]

    def distinct(self, column: str) -> List[str]:
        return sorted({row[column] for row in self.rows})

    def one(self, column: str) -> str:
        found = self.distinct(column)
        if len(found) != 1:
            raise ValueError(f"expected exactly one {column}, got {found}")
        return found[0]

    def __len__(self) -> int:
        return len(self.rows)
