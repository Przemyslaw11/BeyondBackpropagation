# DSS figure set — G1 + PPAM only

**No W1 data.** These figures are built only from G1 (2026-10) and
PPAM-era (2026-09) tables. The script refuses any path containing `w1`
or `results/runs`.

Every ex-M0 number is an **estimate**, marked on the figure:

- energy (Protocol B only): measured MF-Seq energy × (1 − measured PPAM-era M0 share);
- accuracy: assumed unchanged (M0 is dead compute);
- passes: no ex-M0 estimate (G1 has no per-stage epoch counts).

| figure | content |
|---|---|
| `dss_fig1_protocol_a_energy` | Protocol A (G1) accuracy vs energy, four measured arms (no ex-M0) |
| `dss_fig2_protocol_a_passes` | Protocol A (G1) accuracy vs passes, measured arms only |
| `dss_fig3_protocol_b_energy` | Protocol B (PPAM) accuracy vs energy, BP + MF-Seq + ex-M0 estimate |

## M0 shares used

| dataset | M0 Wh (mean) | trace Wh (mean) | M0 share |
|---|---|---|---|
| CIFAR-10 | 3.986 | 17.276 | 0.2286 |
| CIFAR-100 | 3.567 | 15.599 | 0.2275 |

## Inputs (SHA-256)

| file | SHA-256 |
|---|---|
| `pareto_protocol_a.csv` | `b12411d2b8a602bc1a29526f7459c1b1dfe1de02ec3ee8b60ef4ad3f869ff665` |
| `pareto_protocol_b.csv` | `46fdbd0a26acbd66538ed433aab240b920b17ee7584b70efc25aa0e48f1ac9d1` |
| `w0b_stage_energy.csv` | `f9b7de35ad8d6797fa5da4394b7d440072ca4391aa4c377e1f6f5e02d94aec25` |

## Era caveat

Protocol B is PPAM 2026-09 and its M0 share is from the same era, so
the only ex-M0 estimate is drawn there. Protocol A (G1 2026-10) is
shown as measured arms only, to avoid mixing eras.
