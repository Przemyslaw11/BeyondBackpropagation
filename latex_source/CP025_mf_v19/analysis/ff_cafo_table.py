"""Write the complete per-configuration FF and CaFo table (Section 5.6, reviewer
R3b) to tables/ff_cafo_full.csv. Reads only this folder's numbers.json, which
analyse.py produces from artifacts/tidy/runs.csv; no value is typed by hand."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = json.loads((HERE / "numbers.json").read_text())
METRICS = [("test_accuracy", "acc_pct"), ("total_gpu_energy_wh", "energy_wh"),
           ("training_duration_sec", "time_s"), ("peak_torch_alloc_mib", "alloc_mib"),
           ("trace_mean_power_w", "power_w")]


def main() -> None:
    out = HERE.parent / "tables" / "ff_cafo_full.csv"
    cols = ["configuration", "arm", "n"]
    for _, short in METRICS:
        cols += [short + "_mean", short + "_sd"]
    cols += ["acc_delta_vs_bp_pp", "energy_x_bp", "time_x_bp", "alloc_x_bp"]
    rows = []
    for config, arms in NUM["ff_cafo"].items():
        for arm, m in arms.items():
            row = {"configuration": config, "arm": arm, "n": m["n"]}
            for key, short in METRICS:
                row[short + "_mean"] = round(m[key]["mean"], 4)
                row[short + "_sd"] = round(m[key]["sd"], 4)
            if arm != "BP":
                row["acc_delta_vs_bp_pp"] = round(m["test_accuracy"]["delta_vs_bp"], 3)
                row["energy_x_bp"] = round(m["total_gpu_energy_wh"]["ratio_vs_bp"], 3)
                row["time_x_bp"] = round(m["training_duration_sec"]["ratio_vs_bp"], 3)
                row["alloc_x_bp"] = round(m["peak_torch_alloc_mib"]["ratio_vs_bp"], 3)
            rows.append(row)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
