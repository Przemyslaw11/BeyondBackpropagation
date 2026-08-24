# Test Suite Layout

The suite is organized by scope; **selection is still driven by pytest markers**
(`slow`, `gpu`, `smoke`), not by directory. The standard commands
(`make test-fast`, `make smoke`, `make check`) behave identically regardless of
layout.

```
tests/
├── unit/          # Pure in-memory unit tests (math kernels, config validation,
│                  # loaders with fakes, registries, primitives, backend policy)
├── integration/   # Multi-component wiring (ExperimentRunner + adapters,
│                  # lifecycles, infrastructure adapters, CLI wrappers, tuning)
├── regression/    # Characterization & scientific-regression guards
│                  # (architecture/config characterization, seed/FP regressions)
├── smoke/         # Tiny offline end-to-end checks (marked `smoke`)
└── fixtures/      # Static fixture data (baseline artifacts & docs)
```

## Conventions

- Keep markers authoritative: `-m "not slow and not gpu"` must always select a
  CPU-only, fast subset. New slow/gpu tests must carry the corresponding marker.
- Test module basenames must stay unique across directories (pytest default
  import mode).
- Tests import the canonical package (`beyond_backprop.*`); the legacy `src.*`
  namespace was removed (decision MIG-004), so no test may import from `src.*`.
