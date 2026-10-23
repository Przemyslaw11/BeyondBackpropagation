# Publication hygiene — JPDC extension of the PPAM 2026 paper

Status: notes for the JPDC submission, 2026-10-24. Sources were retrieved on
2026-10-23/24 unless marked otherwise. This is a compliance checklist, not
legal advice.

## 1. The signed Springer Nature licence is the controlling document

The PPAM 2026 paper was accepted under the Springer Nature **Licence to
Publish — Proceedings Papers**, form `ER_Book_ProceedingsPaper_LTP_ST_v.2.0.1
(12_2025)`, signed 5 October 2026. The signed PDF is in the project at
`CP025/CP025_COPYRIGHT.pdf`. The clauses that matter for the JPDC extension
are:

- **Clause 2 (Copyright).** "Ownership of copyright in the Contribution will be
  vested in the name of the Author." Springer receives a broad exclusive
  licence, but copyright remains with the author.
- **Clause 4(b) (Submitted Manuscript).** The author may make the submitted
  manuscript available "at any time and under any terms", including a CC BY
  licence, with an acknowledgement and a link to the Version of Record once
  published.
- **Clause 4(c) (Accepted Manuscript).** The author may post the accepted
  manuscript on a personal, self-maintained website immediately on acceptance,
  and in employer/institutional/funder repositories after a **12-month embargo**
  from first publication. Accepted manuscripts may not be shared under a
  Creative Commons licence.
- **Clause 4(d)(v) (Expanded version).** The author may "publish an expanded
  version of their Contribution provided the expanded version (i) includes at
  least 30% new material (ii) includes an express statement specifying the
  incremental change in the expanded version (e.g., new results, better
  description of materials, etc.)."

**Consequence for this submission.** The JPDC article must contain at least 30%
new material and an explicit statement of what is new. `overlap_ledger.md`
targets 63% new words; that is a planning margin, not proof. The final
manuscript must repeat the express statement in the Introduction and cover
letter. The 30% figure comes from the **Springer licence**, not from JPDC.

## 2. JPDC / Elsevier expectations

### 2.1 Submission declaration

The JPDC Guide for Authors states, under "Submission declaration":

> When authors submit an article to an Elsevier journal it is implied that:
> The work described has not been published previously except in the form of a
> preprint, abstract, published lecture, academic thesis, or registered report.
> ... The article is not under consideration for publication elsewhere. ...
> If accepted, the article will not be published elsewhere in the same form,
> in English or in any other language, including electronically, without the
> written consent of the copyright holder.

Source: <https://www.sciencedirect.com/journal/journal-of-parallel-and-distributed-computing/publish/guide-for-authors>
(retrieved 2026-10-24).

The same guide says JPDC "publishes original research papers and timely review
articles" and gives a Regular Paper guideline of 16 double-column pages
(20 single-column pages). It requires a declaration of generative AI use, a
data statement, and (for this journal) Option C research-data instructions:
deposit the data in a relevant repository and cite/link it, or explain why that
is not possible.

### 2.2 What "published lecture" does and does not settle

A conference proceedings paper is plausibly a "published lecture" within the
JPDC declaration. The guide does **not** state a percentage of new material for
extended conference papers, and it does not say that a proceedings paper is
automatically excluded. The safe reading is:

1. disclose the PPAM paper and both arXiv versions at submission;
2. cite the PPAM paper and state the incremental contribution;
3. ensure the new material is substantial (the Springer licence requires at
   least 30%);
4. do not submit the same manuscript elsewhere;
5. expect the journal's similarity screening to flag overlap, and be ready to
   explain it.

### 2.3 Elsevier publishing ethics

Elsevier's publishing-ethics policy states:

> The authors should ensure that they have written entirely original works, and
> if the authors have used the work and/or words of others, that this has been
> appropriately cited or quoted and permission has been obtained where
> necessary.

and lists "plagiarism or undisclosed text re-use" and "undisclosed duplicate or
redundant submission" among the grounds for rejection.

Source: <https://www.elsevier.com/about/policies-and-standards/publishing-ethics>
(retrieved 2026-10-24).

It also says, under multiple/redundant publication:

> In general, an author should not submit for consideration in another journal
> a paper that has been published previously, except in the form of an abstract
> or as part of a published lecture or academic thesis or as an electronic
> preprint.

The phrase "published lecture" is the clause that must be disclosed and
explained, not hidden. The 30% condition is stricter than the Elsevier general
policy and comes from the Springer licence.

## 3. The existing arXiv versions

Two arXiv records are relevant:

| arXiv | Title | Relationship | Evidence |
|---|---|---|---|
| [2511.01061v2](https://arxiv.org/abs/2511.01061) | *Energy-Efficient Deep Learning Without Backpropagation: A Rigorous Hardware-Validated Benchmarking Study of Forward-Only Algorithms* | Same title and authors as the PPAM 2026 paper; v2 updated 2026-10-12. Treat as the PPAM preprint unless the authors confirm otherwise. | arXiv API entry; title/authors match `CP025.tex`. |
| [2509.19063v1](https://arxiv.org/abs/2509.19063) | *Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training* | Earlier, broader preprint (23 September 2025); overlapping results but a different scope. Must be disclosed and cited if any material is reused. | arXiv API entry. |

The arXiv non-exclusive licence (<https://arxiv.org/licenses/nonexclusive-distrib/1.0/>)
says the submitter grants arXiv "a perpetual, non-exclusive license to
distribute this article" and certifies the right to grant that licence. It does
not transfer copyright to arXiv. The authors therefore retain copyright in the
preprints, subject to the Springer licence for the proceedings version.

Elsevier's article-sharing policy says authors "can share their preprint
anywhere at any time"; if the article is accepted, authors are encouraged to
link from the preprint to the published article, and may update the preprint
with the accepted manuscript. Source:
<https://www.elsevier.com/about/policies-and-standards/sharing>
(retrieved 2026-10-24).

**Action.** Cite both arXiv records in the JPDC manuscript and cover letter,
with a one-sentence explanation of how each differs from the submitted article.
Do not describe the JPDC article as unpublished if the arXiv versions are
public; describe it as the expanded, peer-reviewed version.

## 4. Springer self-archiving and reuse

Springer Nature's general self-archiving page says that, for subscription
articles, the accepted manuscript may be self-archived on the author's personal
website and in funder/institutional repositories "after an embargo period".
Source:
<https://www.springer.com/gp/open-access/publication-policies/self-archiving-policy>
(retrieved 2026-10-24).

For this paper the signed proceedings licence is more specific and controls:
personal website immediately; institutional/funder repository after 12 months;
no CC licence on the accepted manuscript. The expanded-version clause
4(d)(v) contains no separate embargo, but it requires the 30% new material and
the express incremental statement. If the JPDC article reuses any PPAM text or
graphics beyond that clause, request permission through Springer Nature's
RightsLink service:
<https://www.springer.com/gp/rights-permissions/obtaining-permissions/882>
(retrieved 2026-10-24).

**Practical rule.** Replot every figure from the underlying data rather than
reusing the PPAM PDF. If a figure or table is retained verbatim, record it in a
sentence-level reuse table and obtain permission before submission.

## 5. JPDC self-overlap: what is actually required

There is no JPDC-specific "30% rule" in the sources located. The requirements
that can be stated with evidence are:

1. **Disclose** the PPAM paper and both arXiv versions.
2. **Cite** the PPAM paper and state the incremental change.
3. **Do not submit** the same manuscript elsewhere; the JPDC declaration
   requires that the article is not under consideration elsewhere.
4. **Make the new material substantial.** The Springer licence imposes at
   least 30% new material for the expanded version. JPDC's general policy
   requires original work and prohibits undisclosed text reuse.
5. **Expect screening.** JPDC says it may check manuscripts with screening
   tools. A similarity report that flags the PPAM paper is not automatically
   disqualifying if the overlap is cited and the new material is documented.

The JPDC Guide for Authors does not state a different percentage, and the
phrase "published lecture" is not defined there. That is the main unresolved
interpretive point; it should be handled by disclosure, not by silence.

## 6. Submission checklist

- [ ] Put the express incremental statement in the Introduction and cover
      letter (draft in `overlap_ledger.md` §4).
- [ ] Cite PPAM 2026 and arXiv:2511.01061v2 and arXiv:2509.19063v1.
- [ ] Regenerate the overlap ledger from the final text; keep it above 30% new
      material under a conservative count.
- [ ] Replace all `\wone{...}` placeholders only after the frozen W1 analysis
      is complete.
- [ ] Replot figures from data; do not extract graphics from the Springer PDF.
- [ ] Run a similarity check against the PPAM camera-ready and both arXiv
      versions; document every match.
- [ ] Add the JPDC data statement (Option C) and the generative-AI declaration
      if applicable.
- [ ] Confirm the class-account acknowledgement before naming
      `plglscclass26-gpu-a100`; see `HANDOFF.md` and the academic-grant note.
- [ ] Confirm the final author list, CRediT roles, funding statement, and
      competing-interest declaration.

## 7. What could not be verified

- The JPDC Guide for Authors does not define "published lecture" or give a
  percentage for extended conference papers. The general Elsevier policy and
  the Springer licence were used to set the conservative rule above.
- The Springer LNCS web page is dynamic and did not render its full text in
  this environment; the signed licence PDF was used instead. That PDF is the
  controlling document for this paper.
- Springer's general self-archiving page gives an "embargo period" without a
  single universal length; the signed licence's 12-month accepted-manuscript
  embargo was used. The expanded-version clause itself states no embargo.
- The exact JPDC similarity threshold is not public. The plan assumes
  disclosure plus substantial new material, not a numeric target.
- The class-account permitted-use question is separate from publication
  hygiene and is recorded in `HANDOFF.md`; it is not resolved here.
