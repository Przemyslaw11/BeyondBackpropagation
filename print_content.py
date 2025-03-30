#!/usr/bin/env python3

import os
import sys


def read_gitignore(gitignore_path):
    """Read .gitignore file and return a list of patterns to ignore."""
    if not os.path.exists(gitignore_path):
        return []

    with open(gitignore_path, "r") as f:
        # Strip whitespace and filter out empty lines and comments
        patterns = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    return patterns


def should_ignore(file_path, ignore_patterns, script_path):
    """Check if a file should be ignored based on .gitignore patterns or being the script itself."""
    # Always ignore the script itself
    if os.path.abspath(file_path) == os.path.abspath(script_path):
        return True

    # Check against .gitignore patterns
    for pattern in ignore_patterns:
        # Simple exact match
        if pattern == file_path:
            return True

        # Simple directory match (pattern ends with /)
        if pattern.endswith("/") and file_path.startswith(pattern):
            return True

        # Simple wildcard match (pattern starts with *)
        if pattern.startswith("*") and file_path.endswith(pattern[1:]):
            return True

        # Specific directory match
        if pattern in file_path.split("/"):
            return True

    return False


def print_file_content(files_list, script_path):
    """Print content of each file in the list, respecting the rules."""
    # First read .gitignore to get patterns to ignore
    ignore_patterns = (
        read_gitignore(".gitignore") if os.path.exists(".gitignore") else []
    )

    for file_path in files_list:
        # Skip if the file doesn't exist
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Skip if the file should be ignored
        if should_ignore(file_path, ignore_patterns, script_path):
            print(f"Skipping {file_path} (ignored by rules)")
            continue

        # Print file path and content
        try:
            with open(file_path, "r") as f:
                content = f.read()

            print(f"\n{'-' * 80}")
            print(f"File: {file_path}")
            print(f"{'-' * 80}")
            print(content)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")


def main():
    # Get the path of this script for self-exclusion
    script_path = os.path.abspath(sys.argv[0])

    # List of files to print (from VSCode tabs)
    files = [
        ".gitignore",
        "scripts/run_experiment.py",
        "requirements.txt",
        "src/utils/config_parser.py",
        "src/utils/helpers.py",
        "src/utils/metrics.py",
        "src/utils/logging_utils.py",
        "src/utils/monitoring.py",
        "src/utils/profiling.py",
        "src/data_utils/preprocessing.py",
        "src/data_utils/datasets.py",
        "src/architectures/ff_mlp.py",
        "src/architectures/cafo_cnn.py",
        "src/architectures/mf_mlp.py",
        "src/architectures/__init__.py",
        "src/algorithms/ff.py",
        "src/algorithms/cafo.py",
        "src/algorithms/mf.py",
        "src/algorithms/__init__.py",
        "src/baselines/bp.py",
        "src/baselines/__init__.py",
        "src/training/engine.py",
        "src/tuning/optuna_objective.py",
        "scripts/run_optuna_search.py",
        "configs/ff/fashion_mnist_mlp_4x2000.yaml",
        "configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml",
        "configs/cafo/fashion_mnist_cnn_3block.yaml",
        "configs/bp_baselines/fashion_mnist_cnn_3block_bp.yaml",
        "configs/cafo/cifar10_cnn_3block.yaml",
        "configs/base.yaml",
    ]

    print_file_content(files, script_path)


if __name__ == "__main__":
    main()
