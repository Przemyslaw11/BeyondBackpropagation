#!/usr/bin/env python3
"""Parse per-task SLURM stdout files (`BBP_G1*.out`) into a stage-anatomy table.

The shared append-only run log (`results/<exp>/<exp>_run.log`) lost header and
mid-block lines during the 2026-10-17 quota window (verified: MF-Seq C100 seed42
shows W1_M1=100 there while the task's own `.out` and the run JSON both show 81,
sum 185). The per-task stdout is written by SLURM itself, one file per array
task, so it does not suffer the shared-writer problem and is the best surviving
source for per-stage epochs, cap binding and early-stopping anatomy.

Extracts per file:
- config path and experiment name (from "Config (i): ...")
- seed (from "Seed: N")
- per-stage epochs for MF arms: M0 from "Finished training for Layer_M0 after N
  epochs."; W1_M1/W2_M2/W3_M3 from "Early Stopping at Epoch N" or, when a stage
  hits the cap, from the degenerate "Early Stopping at Epoch <cap>" (logged when
  early stopping triggers exactly at the cap) / absence of the line is reported
  as `null` (stage never trained or file truncated)
- total epochs for MF ("Finished all layer-wise MF training. Total Epochs (Sum): N")
- final test accuracy ("Test Set Results: Acc: XX.XX%")
- first timestamp line (wall-clock start) and .out mtime

Output: one row per .out file; selection of the final attempt per (experiment,
seed) is done downstream (scripts/jpdc/w0c_cap_anatomy.py) by matching the test
accuracy to the run JSON and keeping the latest start time.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

CONFIG_RE = re.compile(r"Config \((\d+)\): (\S+)")
SEED_RE = re.compile(r"Seed: (\d+)")
M0_RE = re.compile(r"Finished training for Layer_M0 after (\d+) epochs\.")
STAGE_ES_RE = re.compile(r"Layer_(W\d_M\d): Early Stopping at Epoch (\d+)")
TOTAL_RE = re.compile(r"Total Epochs \(Sum\): (\d+)")
ACC_RE = re.compile(r"Test Set Results: Acc: ([\d.]+)%")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")

ARRAY_TASK_RE = re.compile(r"^(BBP_G1[A-Za-z]*)-(\d+)_(\d+)\.out$")


def parse_out(path: Path) -> dict | None:
    config = seed = None
    m0 = None
    stages: dict[str, int] = {}
    total = None
    acc = None
    start = None
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if start is None:
            m = TS_RE.match(line)
            if m:
                start = m.group(1)
        m = CONFIG_RE.search(line)
        if m and config is None:
            config = m.group(2)
        m = SEED_RE.search(line)
        if m and seed is None:
            seed = int(m.group(1))
        m = M0_RE.search(line)
        if m:
            m0 = int(m.group(1))
        m = STAGE_ES_RE.search(line)
        if m:
            stages[m.group(1)] = int(m.group(2))
        m = TOTAL_RE.search(line)
        if m:
            total = int(m.group(1))
        m = ACC_RE.search(line)
        if m:
            acc = float(m.group(1))
    if config is None or seed is None:
        return None
    match = ARRAY_TASK_RE.match(path.name)
    job_id, task_id = (match.group(2), match.group(3)) if match else (None, None)
    experiment = Path(config).stem
    return dict(
        out_file=path.name,
        job_id=job_id,
        task_id=task_id,
        experiment=experiment,
        config=config,
        seed=seed,
        start=start,
        m0_epochs=m0,
        w1_epochs=stages.get("W1_M1"),
        w2_epochs=stages.get("W2_M2"),
        w3_epochs=stages.get("W3_M3"),
        total_epochs=total,
        test_accuracy=acc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outs_dir", type=Path, help="directory with BBP_G1*.out files")
    parser.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = parser.parse_args()

    rows = []
    unparsed = []
    for path in sorted(args.outs_dir.glob("BBP_G1*.out")):
        row = parse_out(path)
        if row is None:
            unparsed.append(path.name)
            continue
        row["out_mtime"] = path.stat().st_mtime
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"parsed {len(df)} files -> {args.out}")
    if unparsed:
        print(f"skipped {len(unparsed)} files without config/seed: {unparsed[:10]}")


if __name__ == "__main__":
    main()
