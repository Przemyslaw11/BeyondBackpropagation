# Verification baseline

Commands run before refactoring:

```text
python3 -m pytest -q                 -> 19 passed
python3 -m compileall -q src scripts -> passed
```

Configuration inventory:

```text
configs/**/*.yaml -> 44 files
```

YAML parsing found one known failure:

```text
configs/tuning/cafo_cifar100_cnn_3block_tune.yaml
  ParserError: expected <block end>, but found '-' at line 30
```

The malformed file and the other known issues are tracked in the decision
register. No large-scale move or deletion was performed during baseline capture.
