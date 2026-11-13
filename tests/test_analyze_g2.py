"""analyze_g2.py on synthetic fixtures: planted effect found, null not found, gaps are incomplete."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.jpdc import analyze_g2 as g2

IDLE_W, SECONDS, SAMPLES = 50.0, 10.0, 100_000


def _write_sweep(directory, pipeline, slope, rng, cells=None, repeats=range(g2.N_REPEATS), drop=None):
    directory.mkdir(parents=True, exist_ok=True)
    cells = cells or [(b, w) for b in g2.BATCHES for w in g2.WIDTHS]
    for ds in g2.DATASETS:
        for r in repeats:
            units = []
            for b, w in cells:
                e_bp = 1e-3 * (w / 2000) ** 2
                for arm in ("bp",) + g2.MF_ARMS:
                    log_a = 0 if arm == "bp" else slope * (math.log(b * w) - math.log(128 * 2000)) + 0.05 + rng.normal(0, 0.01)
                    net = e_bp / math.exp(log_a) * SAMPLES
                    if drop == (ds, r, b, w, arm):
                        continue
                    units.append({"batch": b, "width": w, "arm": arm, "samples": SAMPLES, "seconds": SECONDS,
                                  "counter_energy_j": net + IDLE_W * SECONDS, "trace_energy_j": None})
            rec = {"args": {"dataset": ds, "repeat": r, "pipeline": pipeline},
                   "identity": {"gpu_uuid": f"GPU-{r}"}, "idle_start": {"idle_watts": IDLE_W}, "units": units}
            (directory / f"{ds}_r{r}.json").write_text(json.dumps(rec))


def _write_init(results, rng, shift_pp=0.0, missing_seed=None):
    for ds in g2.DATASETS:
        for exp, offset in ((f"bp_{ds}_mlp_3x2000", 0.0), (f"g2_bp_mfinit_{ds}_mlp_3x2000", shift_pp)):
            d = results / "runs" / exp
            d.mkdir(parents=True, exist_ok=True)
            for seed in g2.SEEDS:
                if exp.startswith("g2") and seed == missing_seed:
                    continue
                acc = 50.0 + seed * 0.01 + offset + (rng.normal(0, 0.1) if exp.startswith("g2") else 0)
                (d / f"{exp}_seed{seed}.json").write_text(json.dumps({"test_accuracy": acc}))


def _run(tmp_path, slope, shift_pp=0.0, missing_seed=None, drop=None):
    rng = np.random.default_rng(1)
    _write_sweep(tmp_path / "sweep", "gpu", slope, rng, drop=drop)
    _write_sweep(tmp_path / "bridge", "host", 0.0, rng, cells=[g2.BRIDGE_CELL])
    _write_init(tmp_path / "results", rng, shift_pp, missing_seed)
    table = g2.main(["--sweep-dir", str(tmp_path / "sweep"), "--bridge-dir", str(tmp_path / "bridge"),
                     "--results-dir", str(tmp_path / "results"), "--out-dir", str(tmp_path / "out")])
    return table.set_index(["hypothesis", "dataset"]).sort_index()


def _row(table, h, ds="cifar100"):
    sub = table.loc[(h, ds)]
    return sub.iloc[0] if isinstance(sub, pd.DataFrame) else sub


def test_planted_effect_is_supported(tmp_path):
    t = _run(tmp_path, slope=0.2)
    assert _row(t, "H-G2a").verdict == "supported"
    assert _row(t, "H-G2a").estimate == pytest.approx(0.2, abs=0.01)
    assert _row(t, "H-G2b").verdict == "supported"
    assert _row(t, "H-G2d").verdict == "supported"  # equal accuracy, small noise: equivalent
    assert _row(t, "H-G2a", "cifar10").verdict == "consistent"
    assert set(t.loc["H-G2c"].verdict) == {"descriptive"}
    assert (tmp_path / "out" / "g2_summary.md").read_text().startswith("# G2 summary")


def test_idle_is_subtracted(tmp_path):
    _run(tmp_path, slope=0.2)
    units = pd.read_csv(tmp_path / "out" / "g2_units.csv")
    bp = units[(units.arm == "bp") & (units.width == 2000)]
    assert np.allclose(bp.net_j_per_sample, 1e-3)


def test_null_slope_and_shifted_init_are_not_supported(tmp_path):
    t = _run(tmp_path, slope=0.0, shift_pp=1.0)
    assert _row(t, "H-G2a").verdict == "not supported"
    assert _row(t, "H-G2d").verdict == "not supported"  # +1 pp is outside the +-0.5 margin


def test_missing_seed_or_cell_is_incomplete(tmp_path):
    t = _run(tmp_path, slope=0.2, missing_seed=50, drop=("cifar100", 2, 8192, 8000, "mf_simul"))
    for h in ("H-G2a", "H-G2b", "H-G2d"):
        assert _row(t, h).verdict == "incomplete"
        assert math.isnan(_row(t, h).p_holm)  # no Holm over an incomplete family


def test_duplicate_repeat_and_wrong_pipeline_are_rejected(tmp_path):
    rng = np.random.default_rng(0)
    _write_sweep(tmp_path / "s", "gpu", 0.1, rng, repeats=[0])
    (tmp_path / "s" / "copy.json").write_text((tmp_path / "s" / "cifar10_r0.json").read_text())
    with pytest.raises(ValueError, match="duplicate"):
        g2.load_sweep(tmp_path / "s", "gpu")
    with pytest.raises(ValueError, match="pipeline"):
        g2.load_sweep(tmp_path / "s", "host")
