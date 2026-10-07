"""Pre-registered statistical analysis for the ablation ladder.

This file is written and committed BEFORE any ladder result is inspected. Every
test, every correction and every equivalence margin below is fixed in advance so
that no analytic choice can be made after seeing which way the numbers fall.

THE LADDER
    1 BP            output_layer readout, no auxiliary losses, global gradients
    2 BP-DS         output_layer readout, auxiliary losses, global gradients
    3 MF-Joint      M_L readout,          auxiliary losses, global gradients
    4 MF-recompute  M_L readout,          auxiliary losses, local  gradients
    5 MF-cache-device  as rung 4, activations cached on the GPU
    6 MF-cache-host    as rung 4, activations cached in host memory

The intended interpretation of adjacent steps is documented below. Step 2 -> 3
changes the readout, parameter ownership, and optimizer family, so it is not a
single-variable contrast:

    1 -> 2  auxiliary supervision
    2 -> 3  readout, extra parameters, and AdamW -> Adam
    3 -> 4  locality: gradients stop at the detach
    4 -> 5  caching activations on device
    5 -> 6  moving that cache off the GPU

PRE-REGISTERED DECISIONS
1.  Primary test is PAIRED BY SEED. Rungs share the seed list, and a seed fixes
    the train/validation split and the initialisation, so the pairing is real
    and removes the split-to-split variance that otherwise dominates.
2.  A Welch unequal-variance test is reported ALONGSIDE every paired test, never
    instead of it. Pairing controls the seed but not the variance structure, and
    the timing and energy variances are grossly unequal across rungs. Where the
    two disagree the Welch result is the conservative one and is what we quote.
3.  Holm-Bonferroni correction WITHIN each metric family. The families are fixed
    here: accuracy, time, energy, memory. Corrections never pool across families.
4.  Bootstrap 95 percent confidence intervals accompany every p-value. 10000
    resamples, paired resampling for the paired test, BCa-free percentile method.
5.  EVERY PARITY CLAIM USES TOST. The equivalence margin for accuracy is fixed at
    0.25 PERCENTAGE POINTS, pre-specified, chosen before seeing the data. A
    non-significant difference test is NOT evidence of equivalence and is never
reported as such: the verdict vocabulary is deliberately restricted to
DIFFERENT, DIFFERENT-BUT-NEGLIGIBLE, EQUIVALENT, and INCONCLUSIVE.

Usage:
    python scripts/analyze_ablation_ladder.py --results-dir results/runs
    python scripts/analyze_ablation_ladder.py --results-dir results/runs --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# --- Pre-registered constants. Do not change these after results are seen. ---

#: Equivalence margin for accuracy parity claims, in percentage points.
ACCURACY_EQUIVALENCE_MARGIN_PP = 0.25

#: Equivalence margin for ratio metrics (time, energy, memory), as a fraction.
#: A 5 percent band; anything inside it is not a saving anyone would notice.
RATIO_EQUIVALENCE_MARGIN = 0.05

#: Two-sided alpha for every test, before multiplicity correction.
ALPHA = 0.05

#: Bootstrap resamples for every confidence interval.
N_BOOTSTRAP = 10000

#: Seed for the bootstrap resampler, so the reported intervals are reproducible.
BOOTSTRAP_SEED = 20260302

#: The rungs, in ladder order. Keys are the algorithm/cache identity of a run.
RUNGS: Tuple[Tuple[str, str, str], ...] = (
    ("bp", "BP", "recompute"),
    ("bp_ds", "BP-DS", "recompute"),
    ("mf_joint", "MF-Joint", "recompute"),
    ("mf_recompute", "MF-recompute", "recompute"),
    ("mf_cache_device", "MF-cache-device", "cache_device"),
    ("mf_cache_host", "MF-cache-host", "cache_host"),
)

#: Metric families. Holm correction is applied within a family, never across.
METRIC_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "accuracy": ("test_accuracy",),
    "time": ("training_duration_sec",),
    "energy": ("total_gpu_energy_wh",),
    "memory": ("peak_torch_alloc_mib",),
}

#: Metrics where a difference is naturally expressed as a ratio, not a gap.
RATIO_METRICS = frozenset(
    {"training_duration_sec", "total_gpu_energy_wh", "peak_torch_alloc_mib"}
)

#: Adjacent-rung contrasts. These decompose the BP -> MF saving step by step.
ADJACENT_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("bp", "bp_ds"),
    ("bp_ds", "mf_joint"),
    ("mf_joint", "mf_recompute"),
    ("mf_recompute", "mf_cache_device"),
    ("mf_cache_device", "mf_cache_host"),
)

#: Contrasts that answer a study question directly.
HEADLINE_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    # Is MF just deep supervision by another name?
    ("bp_ds", "mf_recompute"),
    # The paper's headline: what does the whole ladder buy over BP?
    ("bp", "mf_recompute"),
    ("bp", "mf_cache_device"),
    ("bp", "mf_cache_host"),
)


@dataclass
class BootstrapCI:
    """A percentile bootstrap confidence interval."""

    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.low:.4g}, {self.high:.4g}]"


@dataclass
class ContrastResult:
    """One rung-versus-rung comparison on one metric."""

    metric: str
    family: str
    rung_a: str
    rung_b: str
    n_pairs: int
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float

    paired_p: float
    welch_p: float
    wilcoxon_p: float
    paired_p_holm: float = math.nan
    welch_p_holm: float = math.nan

    difference_ci: Optional[BootstrapCI] = None
    ratio_ci: Optional[BootstrapCI] = None

    tost_p: float = math.nan
    tost_margin: float = math.nan
    equivalent: bool = False

    @property
    def verdict(self) -> str:
        """DIFFERENT, EQUIVALENT or INCONCLUSIVE. Never 'no difference'."""
        # The conservative of the two difference tests decides.
        difference_p = max(self.paired_p_holm, self.welch_p_holm)
        significant = difference_p < ALPHA
        if significant and self.equivalent:
            # Statistically detectable but smaller than the margin we care about.
            return "DIFFERENT-BUT-NEGLIGIBLE"
        if significant:
            return "DIFFERENT"
        if self.equivalent:
            return "EQUIVALENT"
        # Underpowered: we failed to detect a difference AND failed to rule one
        # out. This is the case that must never be dressed up as parity.
        return "INCONCLUSIVE"


@dataclass
class LadderReport:
    """Everything the analysis produces for one experiment configuration."""

    configuration: str
    seeds: List[int]
    rungs_present: List[str]
    contrasts: List[ContrastResult] = field(default_factory=list)
    observed_sd: Dict[str, float] = field(default_factory=dict)


def _bootstrap_ci(
    samples: np.ndarray,
    statistic,
    rng: np.random.Generator,
    n_resamples: int = N_BOOTSTRAP,
) -> BootstrapCI:
    """Percentile bootstrap CI for a statistic of a paired sample matrix.

    Args:
        samples: Array shaped (n_pairs, ...) resampled along the first axis, so
            the pairing between rungs survives every resample.
        statistic: Callable mapping a resample to a scalar.
        rng: Seeded generator, so the interval is reproducible.
        n_resamples: Bootstrap replicate count.
    """
    n = samples.shape[0]
    point = float(statistic(samples))
    if n < 2:
        return BootstrapCI(point=point, low=math.nan, high=math.nan)

    indices = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.array([statistic(samples[idx]) for idx in indices])
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return BootstrapCI(point=point, low=math.nan, high=math.nan)

    low, high = np.percentile(replicates, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return BootstrapCI(point=point, low=float(low), high=float(high))


def _tost_paired(
    differences: np.ndarray, margin: float
) -> Tuple[float, bool]:
    """Two one-sided tests for equivalence on paired differences.

    Returns the larger of the two one-sided p-values (the TOST p-value) and
    whether equivalence is established at ALPHA.

    Equivalence is declared only when we can reject BOTH 'the difference is at
    least +margin' and 'the difference is at most -margin'. Failing to reject a
    difference is not enough and is never treated as enough.
    """
    n = differences.size
    if n < 2:
        return math.nan, False

    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    if sd == 0.0:
        # Identical in every pair: equivalent iff the constant gap is inside the margin.
        return (0.0, abs(mean) < margin)

    se = sd / math.sqrt(n)
    df = n - 1
    # H0_upper: difference >= +margin. Reject if mean is significantly below it.
    t_upper = (mean - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    # H0_lower: difference <= -margin. Reject if mean is significantly above it.
    t_lower = (mean + margin) / se
    p_lower = float(stats.t.sf(t_lower, df))

    tost_p = max(p_upper, p_lower)
    return tost_p, bool(tost_p < ALPHA)


def _holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    finite = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]
    adjusted = [math.nan] * len(p_values)
    if not finite:
        return adjusted

    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running = 0.0
    for rank, (index, p) in enumerate(finite):
        candidate = (m - rank) * p
        # Step-down enforces monotonicity: an adjusted p never decreases.
        running = max(running, candidate)
        adjusted[index] = float(min(1.0, running))
    return adjusted


def _rung_key(record: Dict) -> Optional[str]:
    """Maps a run summary onto its ladder rung, or None if it is not a rung."""
    algorithm = str(record.get("algorithm", "")).lower()
    cache = str(record.get("activation_cache", "recompute")).lower()
    if algorithm in ("bp", "bp_ds", "mf_joint"):
        return algorithm
    if algorithm == "mf":
        return {
            "recompute": "mf_recompute",
            "cache_device": "mf_cache_device",
            "cache_host": "mf_cache_host",
        }.get(cache)
    return None


def _configuration_key(record: Dict) -> str:
    """Groups runs that differ only by rung and seed within one protocol."""
    dataset = str(record.get("dataset", "unknown")).lower()
    architecture = record.get("architecture")
    protocol = str(record.get("protocol", record.get("phase", "main"))).lower()
    return f"{protocol}:{dataset}_{architecture}"


def load_runs(results_dir: Path) -> List[Dict]:
    """Loads every run summary JSON written by the training engine."""
    required_keys = {
        "algorithm",
        "dataset",
        "architecture",
        "seed",
        "test_accuracy",
    }
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping unreadable {path}: {exc}")
            continue
        if record.get("error"):
            print(f"WARNING: skipping failed run {path}: {record['error']}")
            continue
        missing = required_keys - record.keys()
        if missing:
            print(
                f"WARNING: skipping non-run JSON {path}; missing keys: "
                f"{', '.join(sorted(missing))}"
            )
            continue
        records.append(record)
    return records


def _paired_series(
    runs_by_rung: Dict[str, Dict[int, Dict]],
    rung_a: str,
    rung_b: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Extracts the seed-matched metric vectors for two rungs."""
    seeds = sorted(set(runs_by_rung.get(rung_a, {})) & set(runs_by_rung.get(rung_b, {})))
    values_a, values_b, kept = [], [], []
    for seed in seeds:
        a = runs_by_rung[rung_a][seed].get(metric)
        b = runs_by_rung[rung_b][seed].get(metric)
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        values_a.append(a)
        values_b.append(b)
        kept.append(seed)
    return np.array(values_a), np.array(values_b), kept


def _safe_p(value: float) -> float:
    """Maps a degenerate p-value onto 1.0.

    scipy returns nan when both samples have zero variance, which happens when
    two rungs record byte-identical values. That is the strongest possible
    absence of a difference, so it must read as p=1, not as a missing test that
    Holm would then silently drop.
    """
    value = float(value)
    return 1.0 if math.isnan(value) else value


def compare(
    runs_by_rung: Dict[str, Dict[int, Dict]],
    rung_a: str,
    rung_b: str,
    metric: str,
    family: str,
    rng: np.random.Generator,
) -> Optional[ContrastResult]:
    """Runs the full pre-registered battery for one contrast on one metric."""
    a, b, _ = _paired_series(runs_by_rung, rung_a, rung_b, metric)
    if a.size == 0:
        return None

    if metric == "test_accuracy":
        accuracy_values = np.concatenate((a, b))
        if not np.all((accuracy_values >= 0.0) & (accuracy_values <= 100.0)):
            raise ValueError("test_accuracy must be expressed as percentages in [0, 100]")
        if float(np.max(accuracy_values)) <= 1.0:
            raise ValueError("test_accuracy appears to be a fraction, not a percentage")

    if a.size < 2:
        return None

    differences = b - a

    # Paired test: the primary, because the seed fixes split and initialisation.
    paired_p = _safe_p(stats.ttest_rel(b, a).pvalue)
    # Welch: reported alongside, because the variances are badly unequal.
    welch_p = _safe_p(stats.ttest_ind(b, a, equal_var=False).pvalue)
    # Distribution-free backstop for the paired test.
    if np.allclose(differences, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = _safe_p(stats.wilcoxon(b, a).pvalue)

    paired = np.column_stack([a, b])
    difference_ci = _bootstrap_ci(
        paired, lambda s: float(np.mean(s[:, 1] - s[:, 0])), rng
    )

    ratio_ci = None
    if metric in RATIO_METRICS:
        ratio_ci = _bootstrap_ci(
            paired,
            lambda s: float(np.mean(s[:, 1]) / np.mean(s[:, 0]))
            if np.mean(s[:, 0]) != 0
            else math.nan,
            rng,
        )

    # TOST. Accuracy uses the pre-specified 0.25 pp margin; ratio metrics use a
    # margin scaled to the reference rung so 'equivalent' means 'within 5%'.
    if metric in RATIO_METRICS:
        margin = RATIO_EQUIVALENCE_MARGIN * float(np.mean(a))
    else:
        margin = ACCURACY_EQUIVALENCE_MARGIN_PP
    tost_p, equivalent = _tost_paired(differences, margin)

    return ContrastResult(
        metric=metric,
        family=family,
        rung_a=rung_a,
        rung_b=rung_b,
        n_pairs=int(a.size),
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        sd_a=float(np.std(a, ddof=1)),
        sd_b=float(np.std(b, ddof=1)),
        paired_p=paired_p,
        welch_p=welch_p,
        wilcoxon_p=wilcoxon_p,
        difference_ci=difference_ci,
        ratio_ci=ratio_ci,
        tost_p=tost_p,
        tost_margin=float(margin),
        equivalent=equivalent,
    )


def analyze_configuration(
    configuration: str, records: Iterable[Dict], rng: np.random.Generator
) -> LadderReport:
    """Analyses one experiment configuration across all available rungs."""
    runs_by_rung: Dict[str, Dict[int, Dict]] = {}
    seeds = set()
    for record in records:
        rung = _rung_key(record)
        if rung is None:
            continue
        seed = record.get("seed")
        if seed is None:
            continue
        seed = int(seed)
        rung_runs = runs_by_rung.setdefault(rung, {})
        if seed in rung_runs:
            raise ValueError(
                f"Duplicate run for configuration {configuration!r}, "
                f"rung {rung!r}, seed {seed}"
            )
        rung_runs[seed] = record
        seeds.add(seed)

    report = LadderReport(
        configuration=configuration,
        seeds=sorted(seeds),
        rungs_present=[key for key, _, _ in RUNGS if key in runs_by_rung],
    )

    # The two-stage sigma check needs the observed spread of the real runs, not
    # the estimate the power calculation was built on.
    for family, metrics in METRIC_FAMILIES.items():
        for metric in metrics:
            pooled = []
            for rung_runs in runs_by_rung.values():
                values = [
                    float(r[metric])
                    for r in rung_runs.values()
                    if r.get(metric) is not None and math.isfinite(float(r[metric]))
                ]
                if len(values) > 1:
                    pooled.append(np.std(values, ddof=1))
            if pooled:
                report.observed_sd[metric] = float(np.mean(pooled))

    contrasts = list(ADJACENT_CONTRASTS) + [
        pair for pair in HEADLINE_CONTRASTS if pair not in ADJACENT_CONTRASTS
    ]
    for family, metrics in METRIC_FAMILIES.items():
        for metric in metrics:
            for rung_a, rung_b in contrasts:
                result = compare(runs_by_rung, rung_a, rung_b, metric, family, rng)
                if result is not None:
                    report.contrasts.append(result)

    # Holm correction WITHIN each family, never pooled across families.
    for family in METRIC_FAMILIES:
        in_family = [c for c in report.contrasts if c.family == family]
        if not in_family:
            continue
        for attribute in ("paired_p", "welch_p"):
            adjusted = _holm([getattr(c, attribute) for c in in_family])
            for contrast, value in zip(in_family, adjusted):
                setattr(contrast, f"{attribute}_holm", value)

    return report


def _format_report(report: LadderReport) -> str:
    """Renders one configuration's results as text."""
    lines = [
        "",
        "=" * 100,
        f"CONFIGURATION: {report.configuration}",
        f"seeds: n={len(report.seeds)}  rungs: {', '.join(report.rungs_present)}",
        "=" * 100,
    ]

    missing = [key for key, _, _ in RUNGS if key not in report.rungs_present]
    if missing:
        lines.append(f"MISSING RUNGS: {', '.join(missing)}")

    if report.observed_sd:
        lines.append("")
        lines.append("Observed within-rung SD (for the two-stage sigma check):")
        for metric, sd in report.observed_sd.items():
            lines.append(f"  {metric:<28} {sd:.4f}")

    for family in METRIC_FAMILIES:
        in_family = [c for c in report.contrasts if c.family == family]
        if not in_family:
            continue
        lines.append("")
        lines.append(f"--- {family.upper()} (Holm-corrected within this family) ---")
        for c in in_family:
            lines.append(
                f"  {c.rung_a:>16} -> {c.rung_b:<16} n={c.n_pairs:<3} "
                f"{c.mean_a:>10.4g} -> {c.mean_b:<10.4g}"
            )
            lines.append(
                f"      paired p={c.paired_p:.4g} (Holm {c.paired_p_holm:.4g})   "
                f"Welch p={c.welch_p:.4g} (Holm {c.welch_p_holm:.4g})   "
                f"Wilcoxon p={c.wilcoxon_p:.4g}"
            )
            if c.difference_ci:
                lines.append(f"      difference 95% CI: {c.difference_ci}")
            if c.ratio_ci:
                lines.append(f"      ratio      95% CI: {c.ratio_ci}")
            lines.append(
                f"      TOST p={c.tost_p:.4g} at margin {c.tost_margin:.4g}   "
                f"VERDICT: {c.verdict}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-registered analysis of the ablation ladder. Written and "
            "committed before any result was inspected."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Directory of per-run JSON summaries written by the training engine.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the full result set as JSON.",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"No such results directory: {args.results_dir}")

    records = load_runs(args.results_dir)
    if not records:
        raise SystemExit(f"No usable run summaries under {args.results_dir}")

    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        if _rung_key(record) is None:
            continue
        grouped.setdefault(_configuration_key(record), []).append(record)

    if not grouped:
        raise SystemExit("No runs matched a ladder rung.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reports = [
        analyze_configuration(configuration, group, rng)
        for configuration, group in sorted(grouped.items())
    ]

    print(__doc__.split("Usage:")[0].strip())
    for report in reports:
        print(_format_report(report))

    print("")
    print("=" * 100)
    print(
        "REMINDER: INCONCLUSIVE means underpowered. It is not evidence of parity\n"
        "and must not be written up as such. Only an EQUIVALENT verdict, backed by\n"
        "the TOST above, supports a parity claim."
    )
    print("=" * 100)

    if args.json:
        payload = [asdict(report) for report in reports]
        for report_payload, report in zip(payload, reports):
            for contrast_payload, contrast in zip(
                report_payload["contrasts"], report.contrasts
            ):
                contrast_payload["verdict"] = contrast.verdict
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
