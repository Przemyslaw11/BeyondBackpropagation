"""One-shot B2 migration: merge legacy trainer bodies into canonical modules."""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def transform_legacy(text: str) -> str:
    text = text.replace("from src.architectures", "from ..architectures")
    text = re.sub(
        r"from beyond_backprop\.algorithms\.(\w+_math)", r"from .\1", text
    )
    text = text.replace(
        "from src.utils.helpers import",
        "from ..utils.training_support import",
    )
    text = text.replace(
        "from src.utils.logging_utils import log_metrics",
        "from ..utils.training_support import log_metrics",
    )
    text = text.replace(
        "from src.utils.monitoring import get_gpu_memory_usage",
        "from ..utils.training_support import get_gpu_memory_usage",
    )
    text = text.replace(
        "from src.utils.metrics import calculate_accuracy",
        "from ..utils.training_support import calculate_accuracy",
    )
    assert "src.utils" not in text, "untransformed src.utils import"
    assert "src.architectures" not in text, "untransformed src.architectures import"
    return text


def transform_adapter(text: str, train_name: str, eval_name: str) -> str:
    text = text.replace("module = _legacy_module()\n        ", "")
    text = text.replace("module = _legacy_module()\n    ", "")
    text = text.replace(f"module.{train_name}(", f"{train_name}(")
    text = text.replace(f"module.{eval_name}(", f"{eval_name}(")
    text = re.sub(
        r"def _legacy_module\(\) -> Any:\n    return importlib\.import_module\([^)]+\)\n\n\n",
        "",
        text,
    )
    text = text.replace("import importlib\n", "")
    assert "_legacy_module" not in text, "adapter still references _legacy_module"
    return text


SPEC = {
    "ff": ("train_ff_model", "evaluate_ff_model"),
    "mf": ("train_mf_model", "evaluate_mf_model"),
    "cafo": ("train_cafo_model", "evaluate_cafo_model"),
}

for algo, (train_name, eval_name) in SPEC.items():
    legacy = transform_legacy((root / f"src/algorithms/{algo}.py").read_text())
    canonical = (root / f"src/beyond_backprop/algorithms/{algo}.py").read_text()
    adapter = transform_adapter(canonical, train_name, eval_name)
    # Drop the canonical module docstring; the legacy docstring leads the merge.
    adapter_body = adapter.split("\n", 1)[1].lstrip("\n")
    merged = legacy.rstrip() + "\n\n\n" + adapter_body
    (root / f"src/beyond_backprop/algorithms/{algo}.py").write_text(merged)
    print(algo, "merged OK")
