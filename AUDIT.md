# AUDIT.md — JPDC extension, Phase 0.2 + 0.3(a)/(c)

Scope: code at `release/canonical-v2` @ `181ecc4` (working branch `jpdc-extension`), the
PPAM camera-ready configs, the archived run logs and the harmonised ladder table.
Every statement below is [Verified] against code, configs or logs unless marked otherwise.

Sources:
- Code: `src/algorithms/mf.py`, `src/baselines/bp.py`, `src/baselines/bp_ds.py`,
  `src/utils/early_stopping.py`.
- Configs: `configs/base.yaml`, `configs/bp_baselines/*_bp.yaml`, `configs/bp_ds/*`,
  `configs/mf/*`, `configs/mf_joint/*`.
- Logs: `results/*_run.log` (20 seeds per experiment; local copy in `jpdc_data/run_logs/`).
- Tidy table: `artifacts/tidy/runs.csv` (19,944 rows, blob `6f58845`, commit `4a59721`;
  local copies in `jpdc_data/tidy/`).
- Per-unit epoch statistics computed by `scripts/jpdc/parse_run_logs.py`
  (outputs `jpdc_data/tidy/epochs_by_unit.csv`, `multiplier_decomposition.csv`).

## 1. Training paths (what actually runs)

| | BP | BP-DS | MF-Joint | MF (canonical) | MF (legacy, 2025) |
|---|---|---|---|---|---|
| schedule | 1 joint unit | 1 joint unit | 1 joint unit | S sequential stages (M0, W1_M1, W2_M2, W3_M3) | same, no M0 |
| detachment | none | none | none | input detach per stage, `mf.py:572` | same |
| optimizer | AdamW, lr 7.674e-5, wd 3.671e-5 | AdamW, lr 1.46e-4, wd 1.33e-4, aux 0.171 | Adam, lr 1.879e-4, wd 1.67e-6, aux 9.51 | Adam, lr 2.154e-4, wd 1.61e-6, fresh per stage (`mf.py:522`), M0 own optimizer (`mf.py:465`) | same code, other hyperparams |
| early stop | global, val_loss, patience 20, min_delta 0, cap 100 (`bp.py:190`) | global, same policy (`bp_ds.py:251`) | global, same policy | per stage, same policy values (`mf.py:426-440`) | per layer: 15 epochs, patience 4, min_delta 1e-4 (inert `legacy_hyperparameters` block in configs) |
| validation | 1 val pass/epoch | 1 val pass/epoch | 1 val pass/epoch | 1 val local-loss pass per stage per epoch | same |
| checkpoint metric | val_accuracy (config), discarded by `BBP_DISABLE_CHECKPOINTS`; ES still checks val_loss (`bp.py:370`) | n/a | n/a | n/a | n/a |
| M0 stage | — | — | — | trained, default on (`mf.py:449`), feeds neither phase 2 nor evaluation (`mf.py:448`) | not trained |

Notes [Verified]:
- The MF-Joint docstring claim "differs from MF by detachment alone" is true only pairwise
  on the update rule; the canonical MF additionally changes schedule (staged vs joint),
  optimizer state (fresh per stage), ES scope (S stopping rules vs one) and adds M0.
- The comment `mf.py:448` "M0 postdates the published runs" is stale for the PPAM
  canonical logs: `Starting MF training for Layer_M0` appears in 20/20 seeds of every
  canonical MF experiment; only the 2025 legacy protocol skipped M0
  (`configs/reproduction/*_no_m0.yaml` exists for reproducing that variant).
- Canonical ladder MF runs use `activation_cache: recompute` on all four MLP datasets
  (`runs.csv` method column `mf_recompute`); CIFAR also forces recompute because of
  random crop/flip (`configs/mf/cifar10_mlp_3x2000.yaml:30-31`). `cache_device` /
  `cache_host` are separate memory-measurement experiments.
- ES patience produces a deterministic tail of exactly 19 epochs for BP on every dataset
  and seed (best epoch b, stop at b+19) — an off-by-one in the counter that should be
  kept in mind when comparing to "patience 20".

## 2. H1 verdict: MF-Joint → MF confounds detachment with staging — CONFIRMED

The BP→MF-Joint→MF ladder changes five things at once, not one:
1. sequential greedy stages on a frozen prefix vs simultaneous joint updates;
2. fresh optimizer (and Adam state) per stage vs one continuous optimizer;
3. S independent early-stopping decisions (S patience tails) vs one;
4. per-stage epoch cap of 100 vs a single 100-epoch cap;
5. an extra trained unit (M0) that no later stage or evaluation ever consumes.

Quantified from canonical Protocol A logs (mean over 20 seeds; epochs summed over units):

| dataset | BP epochs | MF-Joint | MF | MF/BP |
|---|---|---|---|---|
| CIFAR-10 | 51.95 | 51.30 | 211.2 | 4.07 |
| CIFAR-100 | 42.90 | 50.25 | 191.4 | 4.46 |
| MNIST | 25.40 | 26.85 | 93.3 | 3.67 |
| Fashion-MNIST | 25.96 | 33.95 | 112.2 | 4.32 |

BP-DS for reference: 46.0 (CIFAR-10), 33.25 (CIFAR-100) — the deep-supervision control
does not inflate epochs either.

Conclusion: the 3.7–4.5× epoch multiplier lives almost entirely in the staging
mechanism (S units, per-unit tails, cap resets, M0), not in objective locality.
The clean single-variable test for the paper is therefore MF-Joint + input detach
("MF-Simul"), which keeps the joint schedule and changes only the gradient path.

## 3. Stage-level anatomy of the multiplier (0.3a)

Per-unit epoch statistics, canonical CIFAR-10 MF (means over 20 seeds):

| unit | epochs/unit | ES stops | cap stops (100) | epochs-to-best |
|---|---|---|---|---|
| M0 | 49.5 | 100% | 0% | 29.5 (tail 20.0) |
| W1_M1 | 99.9 | 10% | 90% | not logged |
| W2_M2 | 40.4 | 100% | 0% | not logged (≤ 40.4 − 1) |
| W3_M3 | 21.5 | 100% | 0% | not logged |
| total | 211.2 | | | |

CIFAR-100: M0 44.4, W1_M1 95.1 (50% capped), W2_M2 30.4, W3_M3 21.6 → 191.4.
MNIST: 40.9 + 31.4 + 21.1 → 93.3. Fashion-MNIST: 50.8 + 40.4 + 21.0 → 112.2.

Structural facts:
- The multiplier ≈ the stage count S (4 units for CIFAR nets = M0 + 3 weight layers;
  3 units for the 2-layer MNIST/Fashion nets). Per-stage consumption is close to one
  BP run each (M0 49.5 ≈ BP 51.95 on CIFAR-10).
- Per-stage patience tails add up: each ES'd stage burns ~19–20 extra epochs
  (S × patience vs 1 × patience for BP); M0's tail is exactly 20.
- W1_M1 is the pathological stage: it almost never early-stops before the cap
  (90% capped on CIFAR-10, 50% on CIFAR-100). Its local objective keeps trickling
  improvements so 20 consecutive non-improving epochs rarely accumulate by epoch 100.
- Deep stages converge fast (W3_M3 ~21 epochs total) — "slower convergence" is not the
  general story; unit count + per-unit stop overhead + one cap-bound stage is.
- M0 is pure overhead for accuracy: trained in all canonical runs, consumed by nothing
  (~23% of MF CIFAR-10 epochs).
- Protocol B (`*_equal_epochs`): fixed 13 (CIFAR-10) / 11 (CIFAR-100) epochs per stage →
  52 vs 51.95 and 44 vs 42.9 units-epochs — the matched-pass design reconstructs
  BP's budget from S stages with negligible stopping overhead. This confirms the
  accounting identity; no hidden cost in Protocol B.

Caveat: only M0 logs per-epoch validation curves; W1–W3 emit only start/stop lines,
so their epochs-to-best cannot be computed from run logs. If G1 needs per-stage
best-epochs, recover the W&B offline runs (`~/BeyondBackpropagation/wandb`, 837 MB).
"Epoch" above = one stage-epoch (prefix forward + single-layer update), not a full-net
pass; per-pass cost is folded into the measured NVML/energy aggregates in `runs.csv`.

## 4. Original MF paper vs this repo's MF (schedule mismatch)

Original codebase: Gong, Li, Abdulla, "Mono-Forward: Revisiting Forward-Forward through
Objective-Locality Decomposition", arXiv:2501.09238 (v2, 2026-05-07). Quotes [Verified]:

> "the loss computed from the layer output matrix at layer i is used immediately to
> update the parameters of that layer ... After this local update, the detached layer
> activations are passed forward."

> "MF updates each layer only for one local iteration before the training process
> proceeds to the next layer, well before the current layer has converged."

> (Appendix I) "increasing the number of local iterations K ∈ {1,3,5} ... does not
> yield any improvements in predictive performance". Matched-update protocol: 100/34/20
> epochs for K=1/3/5.

Interpretation: the published MF schedule is a streaming/per-batch scheme — within each
batch every layer performs K local updates in order (K=1 default), so all layers evolve
simultaneously over the same data stream ("deeper layers are trained on representations
produced by earlier layers whose parameters are simultaneously evolving").

The repo's MF instead fully trains each stage to convergence (with early stopping) on a
frozen prefix — i.e., the classical greedy layer-wise paradigm the paper explicitly
contrasts itself with ("Instead of fully training each layer to convergence before
progressing deeper, MF updates each layer only for one local iteration..."). Under
detachment, per-batch in-order updates and simultaneous updates are equivalent up to
floating-point ordering; therefore **MF-Simul (MF-Joint + detach) is actually a closer
reproduction of the published MF training schedule than the repo's staged MF-Seq**.
The repo's variant is the one evaluated by the PPAM camera-ready; this discrepancy
should be stated explicitly in the JPDC extension (and names "MF-Simul"/"MF-Seq"
defined operationally in PREREG.md with this mapping).

## 5. Origin of the "41% less energy / 34% faster" headline (0.3c)

The claim belongs to the Nov 3 2025 arXiv preprint of this work (kb_work.md:210) — same
research thread as the PPAM camera-ready, whose claims were revised to
12.5–19.9% less energy and ≤27.5% less memory (CP025.tex). The protocol difference that
explains it [Assumption, pending arXiv v1 PDF check]: the 2025 runs used the legacy
per-layer schedule (15 epochs/layer, patience 4, min_delta 1e-4), under which MF consumes
about the same number of unit-epochs as BP (CIFAR-10: ~52 vs ~53 measured on the
`repro_*` experiments, ~1.0×) while running cheaper devices per epoch — vs the
camera-ready shared-stopping-policy protocol where MF spends 3.7–4.5× BP's epoch budget.
The `mf_cifar10_mlp_3x2000_legacy_es` and `repro_mf_cifar10_mlp_3x2000` runs exist to
pin this down; complete the two-line trace before quoting it in the paper.

## 6. Power model P = P_static + κ·u (0.3b)

Per-trace fits (OLS on each of the 1024 usable NVML CSVs of 1028; noise-tolerant checkbook
in `scripts/jpdc/fit_power_model.py`, output `jpdc_data/tidy/power_model_fits.csv`):
within-trace slopes are noisy (mean κ 0.28–0.46 W per util-% , mean R² 0.15–0.63) — at 5 Hz
the utilisation signal is bursty and MF in particular operates at a near-constant point
(mean util 6.4% ± narrow range; mean power 60.2 W).

Cross-run fits on trace means (996 runs carrying both `trace_mean_power_w` and
`trace_mean_gpu_util_percent` in the tidy table):

| algorithm | n | fit | R² |
|---|---|---|---|
| BP | 266 | P = 44.9 + 1.66·u | 0.82 |
| BP-DS | 222 | P = 35.3 + 1.97·u | 0.91 |
| CaFo | 40 | P = 54.6 + 1.06·u | 0.81 |
| FF | 28 | P = −10.9 + 2.97·u | 0.99 (extrapolates negative; limited util range) |
| MF | 360 | P = 56.5 + 0.60·u | 0.11 |
| MF-Joint | 80 | P = 45.7 + 1.47·u | 0.87 |
| pooled | 996 | P = 51.2 + 1.14·u | 0.70 |

Observations: BP/BP-DS operate at higher mean utilisation (13.7% / 14.6%) than MF (6.4%)
and draw ~7.5 W more on average — consistent with the paper's "cheaper because it draws
less power, not because it runs faster". Intercepts differ across algorithm groups
(35–57 W), so treat P_static as an operating-regime descriptor, not a pure hardware
constant (confounds: SM clocks, temperature, memory traffic, node mix). MF's own R² is
low because its util span is too narrow to identify a slope; for the Phase 2 cost model
(E2) refit MF with SM-clock as an additional regressor or borrow the slope from the
pooled fit.

## 7. Open items before PREREG completes

1. Confirm from `archive2025/bbp_experiments_2025.tar.gz` and the arXiv v1 (Nov 2025)
   which BP baseline (tuning budget, ES settings) the 41%/34% numbers compared against.
2. Recover W1–W3 per-epoch curves from W&B offline runs if per-stage epochs-to-best
   are needed for the cost model (E2).
3. Data-order/split parity across algorithms: verify identical val split and shuffling
   for a fixed seed (suggested unit test; also validates MF-Simul trajectory test).
4. Confirm the BP model path builds a plain MLP while `configs/bp_baselines/*_bp.yaml`
   still says `model.name: MF_MLP` (naming leftover; check before any G1 rerun of BP).
5. Decide the PREREG primary ES quantity for MF-Simul (proposal: final-readout M_L
   validation CE) and freeze thresholds from observed seed variance.
