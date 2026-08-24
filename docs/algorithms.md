# Algorithms

The algorithm registry exposes BP, FF, CaFo, and MF through the
`TrainingAlgorithm.fit(context)` contract.

- BP retains global cross-entropy optimization with AdamW.
- FF retains local goodness updates, downstream classifier training, and
  algorithm-specific inference.
- CaFo retains block/predictor stages, frozen blocks, stopping, aggregation,
  and inference.
- MF retains layer-wise training, detached preceding activations, local
  validation, projection behavior, and final-layer inference.

The dependency-light FF, CaFo, and MF formulas are available as pure functions
under `beyond_backprop.algorithms.*_math`; legacy training entry points call
those functions during migration.
