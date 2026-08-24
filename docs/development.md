# Development

Install the package with the development extra, then run:

```bash
make check
make smoke
```

The default checks format and lint canonical code and tests, run mypy on the
canonical package, and execute CPU/non-slow tests. Optional GPU, tracking,
tuning, and carbon integrations are separate extras and are not required for
the CPU smoke path.
