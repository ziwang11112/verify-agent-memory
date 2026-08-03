# Paper Build Report

## Receipt

- Built: August 3, 2026.
- Command: `.venv\Scripts\python.exe -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 21.
- Main text ends: page 8.
- References begin: page 8.
- Appendices begin: page 9.
- NeurIPS checklist begins: page 15.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- All 21 pages were rendered at inspection resolution and visually inspected,
  including methodology pages 3--4, experiments and results pages 4--8, the first
  appendix page 9, appendix figures and tables through page 14, and the checklist.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- Figure 1 links the same-memory counterfactual, relevance-by-admissibility matrix,
  and pre-prompt enforcement path. Figure 2 presents the value, source, and
  fragility of metadata constraints in a single mechanism sequence. Figure 3 shows
  the released-oracle/text-inference gap and controlled overflip; Figure 4 shows the
  controlled three-reader paired-exposure effects and residual leakage. The four
  additional audited examples, observational GateMem evidence, and frozen nine-arm
  retrieval diagnostics remain in the appendix. All figures preserve readable
  labels and uncertainty marks, and all generated tables remain within their text
  blocks.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
