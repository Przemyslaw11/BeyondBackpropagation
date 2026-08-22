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
