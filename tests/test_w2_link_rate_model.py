"""Guards for the zero-GPU W2 link-rate accounting model."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "jpdc" / "w2_link_rate_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("w2_link_rate_model", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_communication_energy_is_idle_plus_nic():
    model = _load_module()
    got = model.communication_energy_wh(
        cut_gb=1.0,
        comm_sec=8.0,
        idle_power_w=50.0,
        nic_power_w=2.0,
        nic_pj_per_bit=1.0,
    )
    expected = (50.0 * 8.0 + 2.0 * 8.0 + 1e-12 * 8.0e9) / 3600.0
    assert got == pytest.approx(expected, rel=1e-12)


def test_seq_global_prefix_rates_match_w3():
    model = _load_module()
    assert model.PREFIX_RATE_GB == {1: 0.0, 2: 0.72, 3: 1.44}


def test_paired_ratio_ci_uses_seed_pairs():
    import pandas as pd

    model = _load_module()
    rows = []
    for seed, bp_sec, bp_wh, mf_sec, mf_wh in (
        (42, 10.0, 2.0, 5.0, 1.0),
        (43, 20.0, 4.0, 8.0, 1.6),
    ):
        for arm, sec, wh in (("bp", bp_sec, bp_wh), ("mf_seq", mf_sec, mf_wh)):
            rows.append(
                {
                    "dataset": "cifar100",
                    "arm": arm,
                    "link_mbps": 100.0,
                    "seed": seed,
                    "target_test_accuracy": 30.0,
                    "passes": 10.0,
                    "cut_gb_to_target": 10.0,
                    "compute_sec": 1.0,
                    "communication_sec": sec,
                    "total_sec": sec,
                    "compute_energy_wh": 1.0,
                    "communication_energy_wh": wh,
                    "total_energy_wh": wh,
                    "source": "unit",
                }
            )
    summary = model.summarise(pd.DataFrame(rows))
    row = summary[(summary.arm == "mf_seq") & (summary.link_mbps == 100.0)].iloc[0]
    # Paired ratios are 0.5 and 0.4; their mean is 0.45, not 13/30 = 0.433.
    assert row["total_sec_ratio_vs_bp_mean"] == pytest.approx(0.45)
    assert row["total_sec_ratio_vs_bp_lo"] <= 0.45 <= row["total_sec_ratio_vs_bp_hi"]
