# Metrics and artifacts

Accuracy is represented in percentage points (`0`–`100`). Scalar metrics carry
their unit, source, and measured/estimated provenance through
`MetricValue`.

Canonical runs can persist:

```text
config.resolved.yaml
metadata.json
metrics.json
history.csv
summary.json
checkpoints/
logs/
profiling/
```

The historical `resolved_config.yaml` and `run_summary.json` names are also
written as compatibility aliases. W&B, NVML, and CodeCarbon are optional and
imported lazily. Disabled or unavailable services are represented explicitly;
CPU smoke tests do not require those packages or hardware.
