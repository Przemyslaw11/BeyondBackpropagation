#!/usr/bin/env python3
"""Generate empty W1 table shells from the synthetic fixture tree.

This script does **not** read any W1 run.  It imports the fixture builder from
``tests/test_analyze_w1.py`` (synthetic values, seeds 42-61, no real result),
runs the frozen reducer on it, and keeps only the header rows plus the column
inventory.  The shells document the exact schema the frozen reducer emits, so
the real reduction can be checked against them later.

``scripts/jpdc/analyze_w1.py`` is never modified; its SHA-256 is verified
against the frozen value before and after the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.jpdc import analyze_w1 as reducer  # noqa: E402

FROZEN_ANALYZER = "c424a0eebe81e8e48ec3e3ad5371e9343e6a7650e0ca9d4221c04c74a5a61ff1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixture_module():
    path = REPO / "tests" / "test_analyze_w1.py"
    spec = importlib.util.spec_from_file_location("w1_synthetic_fixture", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load fixture module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "analysis" / "w1_table_shells",
        help="where the header-only shells are written",
    )
    args = parser.parse_args()

    analyzer = REPO / "scripts" / "jpdc" / "analyze_w1.py"
    before = _sha256(analyzer)
    if before != FROZEN_ANALYZER:
        raise SystemExit(
            f"analyze_w1.py hash {before} != frozen {FROZEN_ANALYZER}; refusing"
        )

    fixture = _load_fixture_module()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="w1_shells_") as tmp:
        root = Path(tmp)
        results = fixture._build_tree(root)  # synthetic, deterministic
        # The stock fixture carries no post-training diagnostics, so the
        # secondary tables would have no header.  Add one synthetic block to
        # one run: still fixture data, never a W1 outcome.
        probe_experiment = reducer.EXPERIMENTS["mf_seq"].format(dataset="cifar10")
        probe_run = (
            results / "runs" / probe_experiment
            / f"{probe_experiment}_seed42.json"
        )
        record = json.loads(probe_run.read_text())
        record.update(
            {
                "final_train_loss": 0.01,
                "final_train_accuracy": 99.0,
                "final_val_loss": 0.02,
                "final_val_accuracy": 98.0,
                "final_test_loss": 0.03,
                "final_test_accuracy": 66.0,
                "secondary_eval_seconds": 2.9,
                "secondary_eval_error": None,
                "secondary_eval_scope": "fixture",
                "linear_probe_test_accuracy": {"layer_0": 60.0, "layer_1": 61.0},
                "effective_rank": {"layer_0": 10.0, "layer_1": 11.0},
                "local_head_test_accuracy": {"M1": 60.0, "M2": 61.0},
                "stage_epochs": [
                    {
                        "stage": "M0",
                        "epochs": 10,
                        "stopping_epoch": 10,
                        "max_epochs": 10,
                        "stopped_early": False,
                    }
                ],
            }
        )
        probe_run.write_text(json.dumps(record))
        generated = root / "reduced"
        report = reducer.analyze(results, generated)

        inventory = []
        for csv_path in sorted(generated.glob("*.csv")):
            try:
                frame = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                # e.g. w1_missing_seeds.csv when the fixture has no missing seed
                (out / csv_path.name).write_text("")
                inventory.append(
                    {"table": csv_path.name, "fixture_rows": 0, "columns": []}
                )
                continue
            frame.iloc[0:0].to_csv(out / csv_path.name, index=False)
            inventory.append(
                {
                    "table": csv_path.name,
                    "fixture_rows": len(frame),
                    "columns": list(frame.columns),
                }
            )

        headings = [
            line
            for line in report["summary"].splitlines()
            if line.startswith("#") or line.startswith("##")
        ]
        summary_shell = ["# W1 summary shell (headings only)", ""]
        summary_shell.append(
            "Generated from the synthetic fixture; numbers intentionally removed."
        )
        summary_shell.append("")
        for heading in headings:
            summary_shell.append(heading)
            summary_shell.append("")
            summary_shell.append("<!-- filled by the real reduction -->")
            summary_shell.append("")
        (out / "w1_summary_shell.md").write_text("\n".join(summary_shell))

    after = _sha256(analyzer)
    if after != before:
        raise SystemExit("analyze_w1.py changed during generation; refusing")

    lines = [
        "# W1 table shells",
        "",
        "**SHELLS ONLY — generated from `tests/test_analyze_w1.py` synthetic",
        "fixtures. No W1 run, trace or contrast was read or written here.**",
        "",
        f"- generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- `scripts/jpdc/analyze_w1.py` SHA-256: `{before}` (frozen, unchanged)",
        "- fixture: `tests/test_analyze_w1.py::_build_tree` (synthetic values,",
        "  seeds 42-61, all W1 arms, planted 2x2 effect; no real outcome)",
        "",
        "Each `.csv` in this directory contains the header row and zero data rows.",
        "`w1_summary_shell.md` contains the section headings only.",
        "",
        "| table | fixture rows (not copied) | columns |",
        "|---|---|---|",
    ]
    for row in inventory:
        columns = ", ".join(f"`{name}`" for name in row["columns"])
        lines.append(f"| `{row['table']}` | {row['fixture_rows']} | {columns} |")
    lines.extend(
        [
            "",
            "Regenerate with:",
            "",
            "```bash",
            "PYTHONPATH=. python3 scripts/jpdc/w1_table_shells.py",
            "```",
            "",
            "The real reduction is produced only by the frozen reducer:",
            "",
            "```bash",
            "PYTHONPATH=. python3 scripts/jpdc/analyze_w1.py \\",
            "  --results-dir results --out-dir <w1_out>",
            "```",
        ]
    )
    (out / "SCHEMA.md").write_text("\n".join(lines))
    print(f"wrote {len(inventory)} shells + SCHEMA.md to {out}")
    print(f"analyze_w1.py sha256 {before} (before == after)")


if __name__ == "__main__":
    main()
