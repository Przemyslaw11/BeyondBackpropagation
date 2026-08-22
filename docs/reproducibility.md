# Reproducibility

The canonical runner seeds Python, NumPy, PyTorch, and CUDA when available.
Run metadata records the UTC timestamp, command line, Git revision and dirty
state, Python and PyTorch versions, hostname, device, seed, and configuration
hash.

Dataset downloads are controlled by `data.download`. Tests use patched or
synthetic datasets and never download data. For comparable measurements, keep
the backend, seed, resolved configuration, and monitoring definition fixed.
