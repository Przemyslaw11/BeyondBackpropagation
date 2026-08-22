# Refactoring decision register

This register separates confirmed source behavior from intentional corrections.
Corrections must have a regression test before they become part of the canonical
runtime.

| ID | Decision | Scientific impact | Status |
| --- | --- | --- | --- |
| SCI-001 | Preserve repository paper metadata `arXiv:2511.01061v1`; record the conflicting `2509.19063v1` brief identifier without rewriting citations. | Citation provenance remains auditable. | recorded |
| CFG-001 | Use deterministic base-plus-experiment deep merge with list replacement and explicit CLI overrides. | Resolved configurations become reproducible. | implemented in canonical loader |
| CFG-002 | Repair the malformed CaFo CIFAR100 tuning YAML and normalize `input_channels'`, misplaced CaFo early-stopping keys, and `epochs_per_block` aliases. | No intended algorithmic change; invalid input becomes actionable. | implemented with regression test |
| DATA-001 | Forward `data.download` through the canonical data policy. | Dataset acquisition behavior becomes explicit. | implemented with regression test |
| DATA-002 | Register MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 with explicit channel, image-size, class-count, and split metadata; move preprocessing behind the canonical namespace while retaining a legacy shim. | Dataset identity and transform semantics become inspectable without changing the published protocol. | implemented with regression test |
| ARCH-001 | BP baselines use standard classifier modules; MF BP baselines contain no projection matrices. | Removes algorithm-specific trainable state from BP comparison. | implemented with regression test |
| ARCH-002 | CaFo dummy shape inference runs in evaluation mode and restores module state. | Prevents BatchNorm running-stat mutation during construction. | implemented in fair baseline builder |
| ARCH-003 | Native and fair-baseline architecture construction is selected through explicit registry entries rather than central conditional dispatch. | Architecture parity becomes testable and extensible without changing model mathematics. | implemented with regression test |
| MET-001 | Canonical accuracy fields use percentage points (`0–100`); persisted metrics include units and provenance. | Prevents factor-of-100 interpretation errors. | implemented in contracts |
| MET-002 | Training is the default canonical measurement region; setup and evaluation are excluded. | Efficiency comparisons remain definitionally clear. | recorded |
| BUG-001 | Legacy `run_experiment.py` removes the obsolete CodeCarbon result key. | No scientific effect; avoids misleading cleanup. | implemented with regression test |
| BUG-002 | Initialize cleanup state before `run_training()` enters its `try` block. | Failed setup paths finalize safely. | implemented with regression test |
| OPS-001 | SLURM manifests and config paths are validated before submission. | Prevents silent omissions in batch experiments. | pending |
