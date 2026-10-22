# W1 table shells

**SHELLS ONLY — generated from `tests/test_analyze_w1.py` synthetic
fixtures. No W1 run, trace or contrast was read or written here.**

- generated (UTC): 2026-10-22T15:49:23+00:00
- `scripts/jpdc/analyze_w1.py` SHA-256: `c424a0eebe81e8e48ec3e3ad5371e9343e6a7650e0ca9d4221c04c74a5a61ff1` (frozen, unchanged)
- fixture: `tests/test_analyze_w1.py::_build_tree` (synthetic values,
  seeds 42-61, all W1 arms, planted 2x2 effect; no real outcome)

Each `.csv` in this directory contains the header row and zero data rows.
`w1_summary_shell.md` contains the section headings only.

| table | fixture rows (not copied) | columns |
|---|---|---|
| `w1_2x2.csv` | 24 | `dataset`, `metric`, `effect`, `n`, `mean`, `median`, `ci_low`, `ci_high`, `p` |
| `w1_contrasts.csv` | 40 | `dataset`, `hypothesis`, `contrast`, `metric`, `n`, `mean_diff`, `median_diff`, `ci_low`, `ci_high`, `p`, `ratio_mean`, `ratio_low`, `ratio_high` |
| `w1_controls.csv` | 48 | `dataset`, `control`, `metric`, `n`, `mean_diff`, `median_diff`, `ci_low`, `ci_high`, `p`, `ratio_mean`, `ratio_low`, `ratio_high` |
| `w1_hypotheses.csv` | 10 | `dataset`, `hypothesis`, `status`, `n`, `complete_n20`, `contrast_a`, `delta_a`, `ci_a_low`, `ci_a_high`, `p_a`, `contrast_b`, `delta_b`, `ci_b_low`, `ci_b_high`, `p_b`, `p_raw`, `p_holm`, `cost_a`, `cost_b`, `note` |
| `w1_local_head_per_seed.csv` | 2 | `arm`, `dataset`, `seed`, `head`, `accuracy` |
| `w1_local_head_summary.csv` | 2 | `arm`, `dataset`, `head`, `n`, `accuracy` |
| `w1_missing_seeds.csv` | 8 | `arm`, `dataset`, `missing_seeds`, `n` |
| `w1_per_seed.csv` | 320 | `arm`, `dataset`, `seed`, `test_accuracy`, `test_loss`, `passes`, `energy_wh`, `wall_sec`, `total_run_duration_sec`, `epochs_completed`, `energy_source`, `wall_source` |
| `w1_probe_per_seed.csv` | 2 | `arm`, `dataset`, `seed`, `layer`, `probe_accuracy`, `effective_rank` |
| `w1_probe_summary.csv` | 2 | `arm`, `dataset`, `layer`, `n`, `probe_accuracy`, `effective_rank` |
| `w1_secondary_per_seed.csv` | 1 | `arm`, `dataset`, `seed`, `final_train_loss`, `final_train_accuracy`, `final_val_loss`, `final_val_accuracy`, `final_test_loss`, `final_test_accuracy`, `secondary_eval_seconds`, `secondary_eval_error`, `secondary_eval_scope` |
| `w1_secondary_summary.csv` | 1 | `arm`, `dataset`, `n`, `final_train_loss`, `final_train_accuracy`, `final_val_loss`, `final_val_accuracy`, `final_test_loss`, `final_test_accuracy`, `secondary_eval_seconds` |
| `w1_stage_epochs.csv` | 1 | `arm`, `dataset`, `seed`, `stage`, `epochs`, `stopping_epoch`, `max_epochs`, `stopped_early` |
| `w1_stage_summary.csv` | 1 | `arm`, `dataset`, `stage`, `n`, `epochs`, `stopping_epoch`, `max_epochs` |

Regenerate with:

```bash
PYTHONPATH=. python3 scripts/jpdc/w1_table_shells.py
```

The real reduction is produced only by the frozen reducer:

```bash
PYTHONPATH=. python3 scripts/jpdc/analyze_w1.py \
  --results-dir results --out-dir <w1_out>
```