# Experiment lifecycle

The supported canonical command is:

```bash
python -m beyond_backprop experiment run --config configs/mf/mnist_mlp_2x1000.yaml
```

Use `--dry-run` to resolve configuration, backend, device, seed, and output
policy without loading a dataset or training. Use repeated
`--set section.key=value` arguments for explicit command-line overrides.

The runner finalizes the resource monitor and tracker on success and failure.
Best-state restoration is delegated to the algorithm adapter, so component
specific stopping and checkpoint semantics are not flattened into a shared
epoch loop.

The training measurement region starts after setup and ends before canonical
evaluation. The runner then writes the resolved configuration, runtime
metadata, metrics, event history, summary, checkpoints, logs, and profiling
outputs under the backend result directory. A late tracking or monitoring
failure is reflected in the persisted final status.
