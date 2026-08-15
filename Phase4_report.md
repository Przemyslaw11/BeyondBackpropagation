# Report back — Phase 4

Basis: **109 new runs** in `results/phase4` plus **9** in `results/reproduction`, **19 HPO
studies** complete at their full trial budgets (50 for FF and BP MLP, 30 for BP CNN and
CaFo), **1027** extended monitoring CSVs. `results/runs` untouched at **712**.
`pytest tests/` — 74 passed, 6 subtests passed.

---

## 1. Reproduction check — **MIXED: efficiency reproduces, accuracy does not**

CIFAR-10 3×2000, legacy pre-Phase-2 settings, seeds 7/42/123, n=3, own `run_summary_dir`.

| Claim | Published | Re-run | Verdict |
|---|---|---|---|
| MF faster than BP | 33.8 % | 33.6 % [30.2, 36.0] | **reproduces** |
| MF less energy | 40.7 % | 42.4 % [38.9, 44.8] | **reproduces** |
| MF less peak memory | 5.4 % | 4.6 % | **reproduces** |
| MF epochs | 38.00 ± 0.00 | 38.33 ± 0.58 | **reproduces** |
| MF accuracy edge | **+1.21 pp** | **−0.07 pp** [−0.27, +0.24] | **DOES NOT — underpowered** |

Absolute numbers: BP 62.08 ± 0.58 / 279.3 s / 5.456 Wh / 1167.8 MiB / 54.67 ep;
MF 62.01 ± 0.34 / 185.6 s / 3.143 Wh / 1113.8 MiB / 38.33 ep.

**The pipeline is not silently different — the baseline is stronger.** Our BP reaches
62.08 against the published 61.13 ± 0.35. The published run used seeds 49/50/51; we used
7/42/123. An accuracy edge that disappears under a seed change was a seed artefact, not a
mechanism. The four efficiency claims all land inside CI, so the gate opened.

Two retracted intermediate claims, recorded because both were wrong in the direction that
flattered MF:
- The first attempt showed MF *losing*, traced to `mf_train_m0` defaulting to `True`. The
  published runs are **3-stage**; the 4-stage variant costs 49.00 epochs and 3.977 Wh.
  `results/reproduction_m0mem_bug` keeps those 6 runs as evidence.
- `peak_gpu_mem_used_mib` was recording M0's peak for every MF run (fixed in `6f0a660`).
  I initially attributed the 969 MiB reading to a CUDA-context change; that was **wrong
  and is retracted**. Corrected value 1113.8 MiB vs published 1120 ± 9.

---

## 2. Old versus new hyperparameters — **no two configs collide**

Audited across all 20 re-tuned configs: **`COLLISIONS: none`**. All three pre-existing
collisions were broken by independent search, not by exemption, so the six
`SHARED_HYPERPARAMETER_OPT_OUT` entries are **deleted** and the invariant is now enforced.
The ladder-rung opt-outs remain — rungs 4–6 must share rung 4's parameters by design.

### FF — `ff_lr / ff_wd / downstream_lr / downstream_wd`

| config | old | new |
|---|---|---|
| mnist_mlp_3x1000_ADAMW | 4.759e-4 / 4.169e-4 / 1.073e-2 / 6.153e-3 | 1.332e-3 / 6.031e-4 / 6.707e-3 / 8.298e-3 |
| mnist_mlp_3x1000_SGD | 1.908e-3 / 5.229e-4 / 3.235e-2 / 4.767e-3 | 1.911e-3 / 3.630e-4 / 3.327e-3 / 1.162e-3 |
| mnist_mlp_4x2000 | 5.254e-4 / 4.228e-4 / 2.708e-3 / 3.573e-3 | 7.634e-4 / 3.112e-4 / 3.615e-3 / 2.168e-3 |
| fashion_mnist_mlp_4x2000 | 3.722e-4 / 3.584e-4 / 1.112e-2 / 5.120e-3 | 1.050e-3 / 6.836e-4 / 6.011e-3 / 9.722e-3 |

### BP MLP — `lr / weight_decay`

| config | old | new |
|---|---|---|
| mnist_mlp_3x1000 | 1.329e-4 / 7.114e-4 | 3.114e-4 / 8.406e-5 |
| mnist_mlp_4x2000 | **1.329e-4 / 7.114e-4** ← collided | **1.064e-4 / 6.011e-6** (2.9× apart) |
| fashion_mnist_mlp_4x2000 | 2.041e-4 / 4.187e-6 | 2.367e-4 / 7.372e-6 |

### BP CNN — `lr / weight_decay`

| config | old | new |
|---|---|---|
| mnist_cnn_3block | 1.570e-3 / 6.251e-5 | 2.079e-4 / 2.606e-6 |
| fashion_mnist_cnn_3block | 5.641e-4 / 8.597e-6 | 3.114e-4 / 2.404e-6 |
| cifar10_cnn_3block | 6.763e-3 / 4.742e-6 | 3.418e-3 / 2.732e-6 |
| cifar100_cnn_3block | 3.201e-4 / 5.880e-5 | 2.242e-3 / 6.361e-6 |

### CaFo Rand-CE — `predictor_lr / predictor_weight_decay`

| config | old | new |
|---|---|---|
| mnist_cnn_3block | 4.214e-4 / 1.659e-6 | 3.650e-4 / 2.890e-6 |
| fashion_mnist_cnn_3block | 1.374e-3 / 2.938e-7 | 2.175e-4 / 1.080e-6 |
| cifar10_cnn_3block | 2.411e-4 / 6.358e-6 | 5.647e-4 / 5.711e-6 |
| cifar100_cnn_3block | **2.411e-4 / 6.358e-6** ← collided | **2.623e-4 / 2.223e-5** (2.2× / 3.9×) |

### CaFo DFA-CE — `predictor_lr / predictor_wd / block_lr / block_wd`

| config | old | new |
|---|---|---|
| cafodfa_mnist | 2.539e-4 / 2.084e-6 / 4.978e-4 / 9.594e-7 | 3.465e-4 / 3.479e-5 / 3.914e-4 / 8.620e-7 |
| cafodfa_fashion_mnist | 3.304e-4 / 3.968e-5 / 1.385e-4 / 1.331e-5 | 6.956e-4 / 3.283e-5 / 2.395e-4 / 2.263e-7 |
| cafodfa_cifar10 | 6.678e-4 / 1.570e-5 / 1.374e-4 / 2.938e-7 | 2.505e-4 / 2.873e-7 / 2.932e-4 / 7.044e-6 |
| cafodfa_cifar100 | **identical to cifar10 on all four** ← collided | **3.157e-4 / 2.669e-7 / 1.378e-4 / 4.602e-7** |

The `cafodfa_cifar100` / `cifar10` pair now differs by **15.3×** in `block_weight_decay`
and 2.1× in `block_lr`. The two predictor values remain close (1.3× and 1.08×), which is
worth stating plainly: they were **searched independently and converged**, rather than
copied. That is what the invariant exists to distinguish.

---

## 3. FF's time gap after harmonisation — **halved, and two-thirds of what remains is epoch count**

n=7, seeds 42–48, paired, 10 000-sample bootstrap. Published claim ≈ **13× slower**.

| FF config | end-to-end (per-stage parity) | epochs | **s/epoch (iso-compute)** | **Wh/epoch** |
|---|---|---|---|---|
| mnist_3x1000_adamw | **4.11×** [3.23, 5.14] | 2.66× | **1.55×** | 1.54× |
| mnist_3x1000_SGD | **5.68×** [5.13, 6.18] | 3.82× | **1.49×** | 1.48× |
| mnist_4x2000 | **6.51×** [5.50, 7.52] | 3.37× | **1.93×** | 1.76× |
| fashion_4x2000 | **7.41×** [7.07, 7.79] | 3.78× | **1.96×** | 1.67× |

**Under per-stage parity the gap is 4.1–7.4×; under iso-compute it is 1.5–2.0×.** The
decomposition is exact — end-to-end = epochs × per-epoch — and says FF is only about
half again as expensive per unit of work. Its real disadvantage is needing **2.7–3.8×
more epochs** to converge.

Accuracy versus the matched BP baseline:

| FF config | Δ accuracy (pp) | 95 % CI | verdict |
|---|---|---|---|
| mnist_3x1000_SGD | −0.014 | [−0.193, +0.144] | **UNDERPOWERED** |
| mnist_4x2000 | −0.064 | [−0.219, +0.070] | **UNDERPOWERED** |
| mnist_3x1000_adamw | −1.097 | [−1.294, −0.860] | resolved, FF worse |
| fashion_4x2000 | −0.704 | [−0.874, −0.547] | resolved, FF worse |

Memory: FF peak ratio **1.0085–1.0206×** — FF uses slightly *more*, not less. Note
`peak_gpu_mem_used_mib` is a device-wide NVML reading including the CUDA context;
`peak_torch_alloc_mib` is the honest field.

Incidental but notable: **FF+SGD (98.35) beats FF+AdamW (97.26) by 1.1 pp** on identical
architecture and seeds.

---

## 4. CaFo — **the efficiency story does not survive**

n=5, seeds 42–46, paired, against the re-tuned BP CNN baseline. `mem < 1` = CaFo cheaper.

| config | Δacc (pp) | time | energy | mem | epochs | Wh/ep | s/ep |
|---|---|---|---|---|---|---|---|
| cafo_mnist | −0.34 | 3.93× | 3.03× | **0.933** | 3.99× | 0.76 | 0.98 |
| cafodfa_mnist | −0.34 | 3.61× | 2.91× | 1.011 | 3.04× | 0.96 | 1.19 |
| cafo_fashion | −0.39 | 6.51× | 4.84× | **0.933** | 6.41× | 0.76 | 1.02 |
| cafodfa_fashion | **+0.23** *(underpowered)* | 4.61× | 3.84× | 1.011 | 3.09× | 1.24 | 1.49 |
| cafo_cifar10 | **−18.83** | 3.70× | 3.16× | **0.925** | 3.56× | 0.89 | 1.04 |
| cafodfa_cifar10 | −7.15 | 7.94× | 8.31× | 0.996 | 3.28× | 2.55 | 2.45 |
| cafo_cifar100 | **−16.88** | 7.99× | 7.06× | **0.940** | 7.83× | 0.90 | 1.02 |
| cafodfa_cifar100 | −8.48 | **10.51×** | **10.86×** | 1.014 | 4.21× | 2.62 | 2.53 |

**CaFo never beats BP on time or energy on any dataset**, and loses up to 18.8 pp of
accuracy. The decomposition is clean and the two variants fail differently:

- **Plain CaFo is genuinely cheaper per data pass** (0.76–0.90× Wh/epoch, ~1.0× s/epoch)
  but needs **3.6–7.8× more passes**. Its entire overhead is epoch count.
- **CaFo-DFA is expensive on both axes** on CIFAR (2.55–2.62× Wh/epoch *and* 3.3–4.2×
  more epochs), which is what produces the 10.9× energy figure.

Its one real win is **6–7 % lower peak memory** for the Rand-CE variant, resolved.

Internally, **CaFo-DFA beats plain CaFo by 11.7 pp on CIFAR-10** (78.43 vs 66.75) and
8.4 pp on CIFAR-100 (50.12 vs 41.72) — DFA block training is what makes CaFo viable on
hard data, and is precisely what erases the efficiency claim.

Historical sanity check against the W&B archive (`final/Test_Accuracy`): `cafo_mnist`
98.88 new vs 98.72 median [98.29, 98.84] over n=44; `cafo_fashion` 91.09 vs 89.94
[88.82, 90.81] over n=63. Both land just **above** the historical maximum — consistent
with fresh 30-trial tuning, and showing none of the collapse signature FF exhibited.

---

## 5. W&B archive

**2739 records, 2723 with full history**, manifest rebuilt and consistent. Exceeds the
original 2675 because the export also captured the new Phase 4 runs. 185 records failed
repeatedly with GCS transport errors (concentrated in April–May 2025 runs) and were
restored from the metadata backup, so every run retains its `config` and `summary` —
which is what reproduces the tables. Phase 5's convergence curves are unblocked.

---

## Three things I would not let through to the camera-ready

1. **The FF `val_loss` pathology invalidates earlier FF numbers.** `src/algorithms/ff.py`
   selected checkpoints on `eval_loss` whenever the metric key lacked `"acc"`, so Phase
   2's harmonised `metric: val_loss` restored an **epoch-2 network**. Same config, same
   seed: **89.01 % / 83 s / 22 ep → 97.05 % / 243 s / 65 ep**. Fixed in `9a14332` by
   pinning the four FF experiment configs and four FF tuning configs to
   `val_accuracy`/`max`, budgets untouched. **No FF result produced under the harmonised
   protocol before that commit is valid** — any Phase 2 or Phase 3 artefact quoting FF
   numbers needs re-checking. CaFo was verified structurally immune (it trains and
   evaluates with the same CE on the same predictor logits).

2. **Table 3 must be replaced wholesale, not annotated.** The back-ported CaFo table is
   marked provisional, but the re-measured numbers are far worse for CaFo than what
   currently ships. Leaving a provisional caption over superseded numbers is not enough.

3. **Do not substitute the published BP baseline to recover MF's accuracy edge.** Pairing
   our MF (62.01) against the published BP (61.13) would manufacture a +0.88 pp win out of
   a −0.07 pp result, by taking the two arms from different seed sets and different
   protocols. `results/phase4`, `results/reproduction` and the 2739-record archive all
   contain BP at 62.08; a reproduction package that contradicts its own table is worse
   than a null result. **The honest claim is stronger anyway**: no detectable accuracy
   difference while using **42.4 % less energy, 33.6 % less time and 4.6 % less peak
   memory**, all three resolved. If the comparison must be settled rather than reported
   as underpowered, raise n — at n=3 the interval is roughly ±0.6 pp.

---

## Statistical caution carried forward

Pooled σ = **0.284 pp** (up to 0.43 on 3×2000). n=5 resolves ≈ ±0.35 pp, n=7 ≈ ±0.26 pp.
**Parity cannot be certified at these counts.** Three comparisons contain the null and are
named **underpowered, not equal**: `ff_mnist_3x1000_SGD` (−0.014 pp), `ff_mnist_4x2000`
(−0.064 pp), `cafodfa_fashion` (+0.23 pp).

## Acceptance criteria

| Criterion | Status |
|---|---|
| Reproduction check lands within CI, or phase stops | efficiency yes; accuracy no, reported not suppressed |
| Every config's hyperparameters differ, except recorded opt-outs | **`COLLISIONS: none`**; 6 opt-outs deleted |
| Extended per-run CSV + Wh/epoch and s/epoch | 1027 CSVs (`power_watts`, `gpu_util_percent`, `mem_util_percent`, `gpu_mem_used_mib`, `gpu_temp_celsius`, `sm_clock_mhz`, `compute_processes`, `process_rss_mib`); 0 of 109 summaries missing either per-epoch field |
| No run writes into `results/runs`; ladder stays at 712 | **712** |
| `pytest tests/` green | **74 passed, 6 subtests** |
