# Configuration

Configurations are resolved as:

```text
typed defaults < base YAML < experiment YAML < repeated --set overrides
```

Mappings merge recursively and lists replace the earlier list. Unknown keys,
invalid enum values, incompatible algorithm/model pairs, and invalid stopping
settings fail before data loading.

Validate or inspect a configuration with:

```bash
python -m beyond_backprop validate-config --config CONFIG
python -m beyond_backprop inspect-config --config CONFIG
```

All repository YAML files are checked by the configuration characterization
tests. `data.download` is forwarded explicitly to the dataset constructor.
