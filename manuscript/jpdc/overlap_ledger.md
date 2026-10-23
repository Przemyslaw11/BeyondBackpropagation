# JPDC extension — section map and overlap ledger

Status: planning ledger, 2026-10-24. No W1 outcome was read.
Companion files: `jpdc_skeleton.tex`, `theory_cost_model.tex`,
`publication_hygiene.md`.

## 1. Why this ledger exists

The PPAM 2026 paper (`CP025/CP025.tex`, arXiv:2511.01061v2) and the JPDC
extension share the same experimental programme. The signed Springer Nature
proceedings licence for CP025 (local file `CP025/CP025_COPYRIGHT.pdf`, v.2.0.1,
signed 5 October 2026, clause 4(d)(v)) permits an expanded version **provided**
that it:

1. includes at least 30% new material, and
2. includes an express statement specifying the incremental change.

This ledger is the planning evidence for (1) and the source for the statement
in (2). It is not a legal opinion. The legal test is new *material*, not a word
count; the word budget below is a conservative planning proxy.

## 2. Section map against PPAM 2026

PPAM section word counts are approximate body-text counts from `CP025.tex`
(references, tables, captions and figures excluded). JPDC target counts are
planning budgets for the full manuscript, not final counts.

| JPDC section | PPAM counterpart | Status | Target words | Reused | Revised | New | New share |
|---|---|---|---:|---:|---:|---:|---:|
| Abstract | Abstract | revised | 250 | 0 | 150 | 100 | 40% |
| 1 Introduction | §1 Introduction | revised | 900 | 100 | 400 | 400 | 44% |
| 2 Background and related work | §2 Background and Related Work | revised | 1,000 | 250 | 350 | 400 | 40% |
| 3 End-to-end energy theory | none | **new** | 1,800 | 0 | 0 | 1,800 | 100% |
| 4 Mono-Forward and the cost hypothesis | §3 Mono-Forward and the Cost Hypothesis | revised | 900 | 300 | 350 | 250 | 28% |
| 5 Measuring cost fairly | §4 Measuring Cost Fairly: Two Protocols | revised | 1,200 | 400 | 500 | 300 | 25% |
| 6 W1 pre-registered design | none | **new** | 1,600 | 0 | 0 | 1,600 | 100% |
| 7 Results | §5 Results (subset) | mostly new | 3,200 | 600 | 600 | 2,000 | 63% |
| 8 Discussion and limitations | §6 Discussion and Limitations | revised | 1,200 | 150 | 450 | 600 | 50% |
| 9 Conclusion | §7 Conclusion | revised | 350 | 100 | 150 | 100 | 29% |
| Appendices A–C | none (reproducibility §8 only) | **new** | 700 | 0 | 0 | 700 | 100% |
| **Total** | | | **13,100** | **1,900** | **2,950** | **8,250** | **63.0%** |

**Reused** = close to verbatim from PPAM, with citation.
**Revised** = same material reorganised or rewritten for the journal argument.
**New** = no PPAM counterpart (theory, W1 design, W1 results, appendices).

Summary:

- Reused: 1,900 words (14.5%)
- Revised: 2,950 words (22.5%)
- New: 8,250 words (63.0%)
- New + revised: 85.5%

The 30% floor is therefore exceeded by a wide margin even under a conservative
reading in which some "revised" material is counted as reuse. The final
manuscript must recompute the ledger from the submitted text, because the
budget is not evidence of what was actually written.

## 3. What is genuinely new

The following sections have no PPAM counterpart:

1. **Section 3, end-to-end energy theory.** Defines per-pass energy, pass
   multiplier, the win condition, fixed-cost correction, and testable
   assumptions. The PPAM paper had a cost hypothesis but no formal model.
2. **Section 6, W1 pre-registered design.** Five arms, two datasets, twenty
   paired seeds, tuned BP, scheduled BP, Seq-Global, and Protocol B; the
   freeze and gating rules.
3. **Section 7, W1 results.** All W1 outcome slots are placeholders in the
   skeleton. This is the largest new block.
4. **Appendices.** Protocol deviations, instrumentation error, and full
   per-seed tables.

The PPAM material that remains is the algorithm description, the measurement
protocol, the matched-data-pass result, and the discussion of the earlier
experiments. Those parts are revised and cited, not silently copied.

## 4. Express incremental statement (draft)

> This article is an expanded version of the paper "Energy-Efficient Deep
> Learning Without Backpropagation: A Rigorous Hardware-Validated Benchmarking
> Study of Forward-Only Algorithms" presented at PPAM 2026. The conference
> paper reported the matched-data-pass hardware comparison. This article adds
> (i) a formal end-to-end energy model with a per-pass advantage and pass
> multiplier, (ii) a pre-registered confirmatory family with tuned BP,
> scheduled-BP and budget-matched controls, (iii) the end-to-end and robustness
> analyses, and (iv) the extended validity discussion and reproducibility
> appendices. The new material is identified in the overlap ledger and exceeds
> the 30% threshold required by the Springer Nature proceedings licence.

Place this statement in the cover letter and in the Introduction (the skeleton
already has a version). Do not bury it only in the cover letter.

## 5. Reuse rules

- Do not copy the Springer PDF, LNCS typography, or figure files into the JPDC
  manuscript. Replot figures from the underlying data.
- Any text reused from PPAM must be cited to the PPAM paper, and the PPAM paper
  must be cited in the Introduction and in every result subsection that draws
  on it.
- The arXiv versions must be disclosed and cited; see
  `publication_hygiene.md`.
- If a verbatim block larger than a sentence is retained, record it in a
  sentence-level reuse table and check it against the licence before
  submission.
- Run a similarity check against both arXiv versions and the PPAM camera-ready
  before submission. A high similarity score is not automatically fatal, but
  every match must be explained by citation and the incremental statement.

## 6. Word-budget caveats

- The PPAM body is about 4,300 words by the same rough count. The JPDC target
  is 13,100 words, consistent with the journal's 16 double-column page guide
  (20 single-column pages) for a Regular Paper.
- JPDC requires an abstract of at most 250 words, 1–7 keywords, and 3–5
  highlights of at most 85 characters each. The skeleton's abstract is within
  the limit; the highlights file is still to be written.
- The final ledger must be regenerated after the W1 results are inserted. The
  placeholders are not words and must not be counted as new material.
