#!/usr/bin/env python3

import os
import sys
import fnmatch
import glob


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

    # Always ignore a.txt
    if os.path.basename(file_path) in ["a.txt", "find_slurm.sh"]:
        return True

    relative_path = os.path.normpath(file_path)

    # Check against .gitignore patterns
    for pattern in ignore_patterns:
        # Exact match
        if fnmatch.fnmatch(relative_path, pattern):
            return True

        # Directory match (pattern ends with /)
        if pattern.endswith("/"):
            if relative_path.startswith(pattern) or f"{relative_path}/" == pattern:
                return True

        # Handle patterns with wildcards
        if "*" in pattern or "?" in pattern or "[" in pattern:
            if fnmatch.fnmatch(relative_path, pattern):
                return True
            # Also check directory components
            path_parts = relative_path.split(os.sep)
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        # Handle negation patterns (those starting with !)
        if pattern.startswith("!"):
            if fnmatch.fnmatch(relative_path, pattern[1:]):
                return False

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


def get_config_files():
    """Get all config yaml files based on the directory structure."""
    config_files = []

    # Add base config
    if os.path.exists("configs/base.yaml"):
        config_files.append("configs/base.yaml")

    # Add all config files from subdirectories
    config_dirs = ["configs/bp_baselines", "configs/cafo", "configs/ff", "configs/mf"]

    for config_dir in config_dirs:
        if os.path.exists(config_dir):
            yaml_files = glob.glob(f"{config_dir}/*.yaml")
            config_files.extend(yaml_files)

    return config_files


def discover_files_by_type(
    extensions=[".py", ".yaml", ".md", ".txt", ".slurm"],
    exclude_dirs=["data", "results", "slurm_logs"],
):
    """Discover files with specific extensions, excluding certain directories."""
    discovered_files = []

    for root, dirs, files in os.walk("."):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                # Skip a.txt files
                if file == "a.txt":
                    continue
                discovered_files.append(file_path)

    return discovered_files


def main():
    # Get the path of this script for self-exclusion
    script_path = os.path.abspath(sys.argv[0])

    # Base files to check (non-config files)
    base_files = [
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
    ]

    # Check command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--configs":
            # Only print config files
            files = get_config_files()
        elif sys.argv[1] == "--all":
            # Discover all project files
            files = discover_files_by_type()
        elif sys.argv[1] == "--py":
            # Just Python files
            files = discover_files_by_type(extensions=[".py"])
        elif sys.argv[1] == "--yaml":
            # Just YAML files
            files = discover_files_by_type(extensions=[".yaml"])
        else:
            # Use provided arguments, but filter out a.txt
            files = [f for f in sys.argv[1:] if os.path.basename(f) != "a.txt"]
    else:
        # Default: use base files + config files
        files = base_files + get_config_files()

    print_file_content(files, script_path)


if __name__ == "__main__":
    main()
