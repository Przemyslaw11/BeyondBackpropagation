# JPDC extension manuscript workspace

Zero-GPU drafts, 2026-10-24. No W1 outcome, trace value, or contrast is present
in these files.

| File | Purpose |
|---|---|
| `jpdc_skeleton.tex` | Elsevier `elsarticle` skeleton; section map in comments; every W1 result is a `\wone{...}` placeholder. |
| `theory_cost_model.tex` | Section 3 draft: per-pass energy, pass multiplier, win condition, fixed-cost correction, testable assumptions. Claims marked `[Definition]`, `[Derivation]`, `[Conjecture]`. |
| `overlap_ledger.md` | Section-by-section reuse/revised/new ledger with word budget, 63% new material target, and the express incremental statement. |
| `publication_hygiene.md` | Springer licence clauses, JPDC/Elsevier overlap policy, arXiv versions, self-archiving, and submission checklist. |
| `highlights.md` | Draft JPDC highlights (3–5 bullets, ≤85 characters each). |
| `references.bib` | Bibliography for the skeleton, copied from `CP025.bib` plus the PPAM preprint entry. |
| `jpdc_skeleton.pdf` | Draft build of 2026-10-24 (14 pages, red W1 placeholders). Not a submission. |

The skeleton builds with `pdflatex` + `bibtex` when `elsarticle.cls` is
available; the 2026-10-24 draft compiled to 14 pages. It still needs the W1
table shells and the final results before it is a submission. Do not replace
the placeholders with values until the frozen analysis is complete and the
author has authorized reading the outcomes.

Related frozen artifacts (read-only):

- `analysis/w1_table_shells/` — header-only tables and schema.
- `analysis/dss_figures/` — DSS figures from G1/PPAM only.
- `PREREG-phase2.md` — protocol and dated amendments.
- `HANDOFF.md` — operational state, allocation IDs, copy-out records.
