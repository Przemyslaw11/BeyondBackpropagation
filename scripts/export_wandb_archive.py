"""Archive the historical Weights & Biases runs to local disk.

Phase 4, task 5. Three things depend on this archive and none of them can be met
from the live service:

* The published hardware figures were exported from W&B's own ``system.*`` panels
  rather than from the project's NVML instrumentation. Only the raw history proves
  that, and reviewers were told those traces were NVML.
* The per-epoch convergence curves behind Figures 1 to 4 exist nowhere else. Phase 5
  has to rebuild those figures from their inputs.
* The published table values need a second source to cross-check the reproduction
  check against.

One JSON file per run under ``<out>/runs``, plus an ``index.jsonl`` manifest. Runs
already on disk are skipped, so an interrupted export resumes by re-running.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import wandb

DEFAULT_PROJECT = "przspyra11/BeyondBackpropagation"


def _clean(value: Any) -> Any:
    """json.dump writes bare NaN and Infinity, which are not valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _history(run: Any, keys: List[str], page_size: int) -> List[Dict[str, Any]]:
    """Full history, not the 500-point sampled view ``run.history()`` returns."""
    if not keys:
        return []
    return [_clean(dict(row)) for row in run.scan_history(keys=keys, page_size=page_size)]


def export(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    runs_dir = out_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=args.timeout)
    runs = api.runs(args.project, per_page=args.page_size)
    total = len(runs)
    print(f"{args.project}: {total} runs; writing to {out_root}")

    written = skipped = failed = 0
    started = time.time()

    for i, run in enumerate(runs, start=1):
        if args.limit and i > args.limit:
            break
        if args.match and args.match not in (run.name or ""):
            continue
        target = runs_dir / f"{run.id}.json"
        if target.exists() and not args.force:
            skipped += 1
            continue

        try:
            summary = _clean(dict(run.summary))
            # Metric keys are the evidence for which traces are ours and which
            # are the W&B agent's, so they are archived even when history is not.
            metric_keys = sorted(run.summary.keys())
            system_keys = [k for k in metric_keys if k.startswith("system")]
            record = {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": str(run.created_at),
                "url": run.url,
                "group": run.group,
                "job_type": run.job_type,
                "tags": list(run.tags),
                "config": _clean(dict(run.config)),
                "summary": summary,
                "metric_keys": metric_keys,
                "system_metric_keys": system_keys,
                "runtime_sec": summary.get("_runtime"),
            }
            if not args.no_history:
                scalar_keys = [
                    k
                    for k in metric_keys
                    if not k.startswith("_")
                    and isinstance(summary.get(k), (int, float, type(None)))
                ]
                record["history"] = _history(run, scalar_keys, args.page_size)
            if args.system:
                # The agent writes GPU power, clocks, utilisation and host RSS to
                # a separate stream that scan_history does not reach.
                events = run.history(stream="events", pandas=False)
                record["system_history"] = [_clean(dict(row)) for row in events]

            target.write_text(
                json.dumps(record, indent=None, default=str), encoding="utf-8"
            )
            written += 1
        except Exception as exc:  # one unreachable run must not sink the archive
            failed += 1
            print(f"  [{i}/{total}] {run.id}: FAILED ({exc})", flush=True)

        if i % 50 == 0 or i == total:
            rate = i / max(time.time() - started, 1e-9)
            print(
                f"  [{i}/{total}] written={written} skipped={skipped} "
                f"failed={failed} ({rate:.1f} runs/s)",
                flush=True,
            )

    print(f"done: written={written} skipped={skipped} failed={failed}")
    # Rebuilt from disk rather than from this pass, so a filtered pass does not
    # drop the runs an earlier pass already archived.
    index_path = _write_index(out_root, runs_dir)
    print(f"manifest: {index_path}")


def _write_index(out_root: Path, runs_dir: Path) -> Path:
    index_path = out_root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as index:
        for path in sorted(runs_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            index.write(json.dumps(_manifest(record)) + "\n")
    return index_path


def _manifest(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "name": record["name"],
        "state": record["state"],
        "created_at": record["created_at"],
        "experiment_name": (record.get("config") or {}).get("experiment_name"),
        "algorithm": ((record.get("config") or {}).get("algorithm") or {}).get("name")
        if isinstance((record.get("config") or {}).get("algorithm"), dict)
        else (record.get("config") or {}).get("algorithm"),
        "seed": ((record.get("config") or {}).get("general") or {}).get("seed")
        if isinstance((record.get("config") or {}).get("general"), dict)
        else None,
        "test_accuracy": (record.get("summary") or {}).get("final/Test_Accuracy"),
        "n_history_rows": len(record.get("history") or []),
        "n_system_keys": len(record.get("system_metric_keys") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out", default="results/wandb_archive")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Config, summary and metric keys only. Much faster for a first pass.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Also pull the W&B agent's system event stream (GPU power, clocks, RSS).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download runs already on disk."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after this many runs. 0 means all."
    )
    parser.add_argument(
        "--match",
        default="",
        help="Only export runs whose name contains this substring.",
    )
    export(parser.parse_args())


if __name__ == "__main__":
    main()
