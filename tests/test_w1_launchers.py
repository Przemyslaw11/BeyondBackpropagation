"""Static guards for the split W1 submission manifest (2026-10-22)."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "scripts" / "slurm_scripts"

EXPECTED = {
    "w1_family.slurm": (10, 200),
    "w1_seqglobal.slurm": (2, 40),
    "w1_family_controls.slurm": (4, 80),
    "w1_controls_protocol_b.slurm": (4, 80),
    "w1_bp_sched.slurm": (2, 40),
}


def test_split_arrays_match_config_counts():
    for name, (n_configs, n_tasks) in EXPECTED.items():
        text = (SLURM / name).read_text()
        configs = re.findall(r'^\s+"(configs/[^"]+\.yaml)"', text, re.M)
        array = int(re.search(r"--array=1-(\d+)", text).group(1))
        assert len(configs) == n_configs, name
        assert array == n_tasks == n_configs * 20, name


def test_new_w1_launchers_use_authorized_account():
    for name in EXPECTED:
        text = (SLURM / name).read_text()
        assert "#SBATCH -A plgoncotherapy2-gpu-a100" in text, name


def test_submit_helper_exposes_split_modes():
    text = (SLURM / "w1_submit.sh").read_text()
    for mode in ("core", "controls", "seqglobal", "protocolb", "sched"):
        assert f"  {mode})" in text
    assert 'ACCOUNT="${ACCOUNT:-plgoncotherapy2-gpu-a100}"' in text
    assert 'TEAM="${TEAM:-plggfastsim}"' in text
