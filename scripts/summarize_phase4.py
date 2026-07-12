"""Summarise a Phase 4 campaign and pair each method against its BP comparator.

The ladder analysis in ``analyze_ablation_ladder.py`` is pre-registered and keyed
by rung, so it cannot read Phase 4's method-versus-baseline layout. This does the
same statistics over a flat results directory: seed-paired bootstrap intervals,
and an explicit power statement for anything the seed count cannot resolve.

Nothing here declares equivalence. Pooled sigma of test accuracy on this hardware
is 0.284 pp and reaches 0.43 pp on the 3x2000 MLPs, so n = 5 resolves about
+/-0.35 pp and n = 7 about +/-0.26 pp. A difference smaller than that is
underpowered, which is not the same claim as equal.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ALPHA = 0.05
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260302

# Pooled within-configuration SD of test accuracy measured over the 712-run
# ladder in Phase 3. Used only to state what a seed count can resolve.
POOLED_ACCURACY_SD_PP = 0.284

DIFFERENCE_METRICS = ("test_accuracy",)
RATIO_METRICS = (
    "training_duration_sec",
    "total_gpu_energy_wh",
    "peak_gpu_mem_used_mib",
    "epochs_completed",
    "gpu_energy_wh_per_epoch",
    "training_sec_per_epoch",
)


def load_runs(results_dir: Path) -> Dict[str, Dict[int, Dict]]:
    """Run summaries grouped by experiment name and then by seed."""
    grouped: Dict[str, Dict[int, Dict]] = {}
    for path in sorted(results_dir.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping unreadable {path}: {exc}")
            continue
        if record.get("error"):
            print(f"WARNING: skipping failed run {path}: {record['error']}")
            continue
        name = record.get("experiment_name")
        seed = record.get("seed")
        if name is None or seed is None:
            continue
        grouped.setdefault(name, {})[int(seed)] = record
    return grouped


def _finite(values: Sequence) -> List[float]:
    out = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out.append(float(v))
    return out


def _bootstrap(
    paired: np.ndarray, statistic, rng: np.random.Generator
) -> Tuple[float, float, float]:
    """Percentile CI for a statistic of a seed-paired (n, 2) matrix."""
    n = paired.shape[0]
    point = float(statistic(paired))
    if n < 2:
        return point, math.nan, math.nan
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    reps = np.array([statistic(paired[i]) for i in idx])
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return point, math.nan, math.nan
    low, high = np.percentile(reps, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return point, float(low), float(high)


def _resolvable_pp(n: int) -> float:
    """Half-width of the 95% CI on a mean accuracy at this seed count."""
    if n < 2:
        return math.inf
    from scipy import stats

    return float(stats.t.ppf(1 - ALPHA / 2, n - 1)) * POOLED_ACCURACY_SD_PP / math.sqrt(n)


def describe(records: Dict[int, Dict], metrics: Sequence[str]) -> Dict[str, Dict]:
    out = {}
    for metric in metrics:
        vals = _finite([r.get(metric) for _, r in sorted(records.items())])
        if not vals:
            continue
        arr = np.array(vals)
        out[metric] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        }
    return out


def contrast(
    method: Dict[int, Dict], baseline: Dict[int, Dict], rng: np.random.Generator
) -> List[Dict]:
    seeds = sorted(set(method) & set(baseline))
    results = []
    for metric in DIFFERENCE_METRICS + RATIO_METRICS:
        pairs = []
        for seed in seeds:
            a, b = method[seed].get(metric), baseline[seed].get(metric)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                continue
            a, b = float(a), float(b)
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            pairs.append((a, b))
        if len(pairs) < 2:
            continue
        paired = np.array(pairs)
        if metric in DIFFERENCE_METRICS:
            kind = "difference"
            point, low, high = _bootstrap(
                paired, lambda s: float(np.mean(s[:, 0] - s[:, 1])), rng
            )
            null = 0.0
        else:
            kind = "ratio"
            point, low, high = _bootstrap(
                paired,
                lambda s: float(np.mean(s[:, 0]) / np.mean(s[:, 1]))
                if np.mean(s[:, 1]) != 0
                else math.nan,
                rng,
            )
            null = 1.0
        crosses = (
            math.isnan(low) or math.isnan(high) or (low <= null <= high)
        )
        results.append(
            {
                "metric": metric,
                "kind": kind,
                "n_pairs": paired.shape[0],
                "point": point,
                "low": low,
                "high": high,
                "crosses_null": bool(crosses),
            }
        )
    return results


def _fmt(value: float, width: int = 10, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return " " * (width - 3) + "n/a"
    return f"{value:{width}.{places}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/phase4")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="METHOD=BASELINE",
        help="Experiment names to contrast, e.g. ff_hinton_mnist_mlp_4x2000=bp_mnist_mlp_4x2000",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    grouped = load_runs(Path(args.results_dir))
    if not grouped:
        print(f"No run summaries under {args.results_dir}")
        return

    metrics = DIFFERENCE_METRICS + RATIO_METRICS
    report: Dict[str, object] = {"results_dir": args.results_dir, "groups": {}}

    print(f"=== {args.results_dir}: {len(grouped)} experiments ===\n")
    for name in sorted(grouped):
        stats_by_metric = describe(grouped[name], metrics)
        seeds = sorted(grouped[name])
        print(f"{name}  n={len(seeds)}  seeds={seeds}")
        for metric, s in stats_by_metric.items():
            print(f"   {metric:<26}{_fmt(s['mean'])} +/- {_fmt(s['sd'], 8)}")
        if "test_accuracy" in stats_by_metric:
            print(
                f"   resolves accuracy to +/-{_resolvable_pp(len(seeds)):.2f} pp "
                f"at the ladder's pooled SD"
            )
        print()
        report["groups"][name] = {"seeds": seeds, "metrics": stats_by_metric}

    if not args.pair:
        return

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = []
    print("=== seed-paired contrasts (method vs baseline) ===\n")
    for spec in args.pair:
        method_name, _, baseline_name = spec.partition("=")
        if method_name not in grouped or baseline_name not in grouped:
            print(f"{spec}: missing one side, skipping\n")
            continue
        rows = contrast(grouped[method_name], grouped[baseline_name], rng)
        shared = sorted(set(grouped[method_name]) & set(grouped[baseline_name]))
        print(f"{method_name}  vs  {baseline_name}   paired seeds={shared}")
        print(f"   resolves accuracy to +/-{_resolvable_pp(len(shared)):.2f} pp")
        for row in rows:
            null = "0" if row["kind"] == "difference" else "1"
            verdict = "UNDERPOWERED" if row["crosses_null"] else "resolved"
            print(
                f"   {row['metric']:<26}{row['kind']:<11}"
                f"{_fmt(row['point'])}  [{_fmt(row['low'], 9)},{_fmt(row['high'], 9)}]"
                f"  vs {null}  {verdict}"
            )
        print()
        contrasts.append(
            {"method": method_name, "baseline": baseline_name, "rows": rows}
        )

    print(
        "UNDERPOWERED means the interval contains the null, which is not evidence "
        "that the two are equal."
    )
    report["contrasts"] = contrasts

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
