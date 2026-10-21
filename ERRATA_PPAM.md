# ERRATA_PPAM.md — corrections owed by the JPDC extension to the PPAM record

Status: draft, 2026-10-17. Scope: statements in the PPAM paper record (README "Key Results",
AUDIT-verified ladders) that the JPDC extension's G1 experiment changes. The PPAM paper's
**measured numbers are not retracted**; what moves is the attribution one of them carries — and
one claim is genuinely inverted. All corrections cite `PREREG.md` §10 (2026-10-17 amendment),
`G1_ATTRIBUTION_SUMMARY.md`, and the frozen analysis `scripts/jpdc/analyze_g1.py`.

| # | PPAM record says | Status | Correction (evidence) |
|---|---|---|---|
| 1 | "A pre-specified ablation identifies gradient **detachment**, not MF's auxiliary losses, as the dominant factor tested." (README, In short) | **Inverted reading** | G1 separates detachment from staging: MF-Simul (detachment without staging, tuned like every arm) is **−0.86/−0.27 pp vs BP** on C10/C100 (p<0.001 / 0.076). The dominant factor is MF's **staged schedule** (per-stage optimizers + independent early stopping); detachment alone lands at BP. (`G1_ATTRIBUTION_SUMMARY.md`; PREREG decision rule → verdict B) |
| 2 | "The detach step … is the largest single gain (**+4.10 and +3.18 pp**) … and it also carries essentially the whole energy increase." (README, ablation ladder) | **Re-attributed** | +4.10/+3.18 pp (MF-Joint→MF) *includes* the staging change; it is not attributable to detachment. Detachment alone: −0.86/−0.27 pp and **−0.93 Wh vs BP on C100** (×0.81 passes), i.e. no accuracy gain and an energy *saving* on C100; on C10 it costs +0.50 Wh. MF-Seq's energy increase is dominated by its ×4.4 passes and its dead M0 stage (22.9% of its energy; `w0b_summary.md`); its W1_M1 unit hits the 100-epoch cap in 90%/55% of runs and the per-task checkpoint anatomy matches the PPAM-era profile (`w0c_summary.md`). |
| 3 | MF own convergence: "+3.90 and +5.04 pp … at 3.2× to 4.0× BP's energy because it needs 3.5× to 4.5× as many passes" (README) | **Magnitudes stand; refined** | G1 reproduces +3.90/+5.04 pp at n=20 contemporaneous. Pass multiplier with JSON accounting: **3.98×/4.36×** (the paper margin rounds to the same claim; log-parsed BP counts were low by 1 epoch systematically — see `analyze_g1` amendment). Energy ratio **3.45×/3.72×**. |
| 4 | Protocol B "matched passes" savings (12.5–19.9% energy) attributed to MF | **Stands, scope note** | Measurements unchanged (PPAM-era traces, not affected by the 2026-10-17 incidents). Scope: this is a property of the **staged** MF; the unstaged MF-Simul's saving is C100-only (−20.9% Wh) and reverses on C10 (+9%). Do not generalize "per-pass cheaper" to detached learning per se. |
| 5 | "The pass multiplier, not the cost of a pass, is what stands between MF and an end-to-end energy advantage." (README) | **Stands, sharpened** | Confirmed: the multiplier lives in the schedule, not in detachment; Seq-Global (W1) isolates whether the frozen prefix adds anything on top (`PREREG-phase2.md`). |

## Data-integrity notes (affect the JPDC extension, not the PPAM numbers)

- PPAM-era camera-ready runs were produced 2026-09-10/11; their traces and logs are intact. The
  2026-10-17 quota/shared-log incidents touch **only** the JPDC G1 reruns (12/160 traces lost
  and rebuilt by job 3376703; pass counts moved to per-run JSON). G1's verdict was accuracy-based
  and unaffected either way (`PREREG.md` §10).
- `analyze_g1.py` as cited in `PREREG.md` §10 carries the 2026-10-17 amendment: passes from run
  JSON `epochs_completed`, energy strictly from each run's own `monitoring_csv_path` (no glob
  fallback).

## Required follow-ups before publication

1. Fix the README "In short" and ablation-ladder paragraphs once W1 results land (the corrected
   attribution sentence, one footnote).
2. The JPDC paper cites this file; the arXiv v2 (if produced) must include rows 1–3.
3. DSS/figures deadline 15 Nov: W1 table (2×2 + no-M0 + cap/BP controls) drafted on the frozen
   analysis stack.
