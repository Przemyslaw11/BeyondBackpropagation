#!/usr/bin/env python3
"""Hash the frozen W1 config set before the freeze.

The HPO-gated configs (``bp_sched``, ``mf_global`` and the Protocol-B
planning configs) are deliberately excluded: they are not frozen until their
dated branches are recorded in ``PREREG-phase2.md`` §9.  The frozen set below
is the one that must stay byte-identical for the core and BP-bestval arrays.

Hash definition: SHA-256 over, for each file in sorted relative-path order,
``relative_path UTF-8 || NUL || file_bytes || NUL``.  Paths are repository
relative, so the hash is checkout-location independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FROZEN_CONFIGS = (
    "configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml",
    "configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml",
    "configs/mf/cifar10_mlp_3x2000.yaml",
    "configs/mf/cifar100_mlp_3x2000.yaml",
    "configs/mf_joint/cifar10_mlp_3x2000.yaml",
    "configs/mf_joint/cifar100_mlp_3x2000.yaml",
    "configs/mf_simul/cifar10_mlp_3x2000.yaml",
    "configs/mf_simul/cifar100_mlp_3x2000.yaml",
    "configs/mf_nom0/cifar10_mlp_3x2000.yaml",
    "configs/mf_nom0/cifar100_mlp_3x2000.yaml",
    "configs/bp_bestval/cifar10_mlp_3x2000.yaml",
    "configs/bp_bestval/cifar100_mlp_3x2000.yaml",
)


def manifest(repo_root: Path) -> dict:
    files = []
    combined = hashlib.sha256()
    for relative in FROZEN_CONFIGS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        combined.update(b"\0")
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    return {
        "hash_algorithm": "sha256(path\\0bytes\\0 ...)",
        "config_set_sha256": combined.hexdigest(),
        "n_files": len(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = manifest(args.repo_root.resolve())
    print(result["config_set_sha256"])
    for row in result["files"]:
        print(f"{row['sha256']}  {row['path']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
