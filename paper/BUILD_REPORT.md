# Paper Build Report

## Receipt

- Built: August 2, 2026.
- Command: `.venv\Scripts\python.exe -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 21.
- Main text ends: page 9.
- References begin: page 9.
- Appendices begin: page 10.
- NeurIPS checklist begins: page 14.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- The revised methodology pages 3--4, experiment and results pages 5--9, first
  appendix page 10, and appendix figures and tables through page 14 were rendered at inspection
  resolution and visually inspected.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- Figure 1's four audited query-memory cases, use/drop decisions, and compact
  verification trace remain legible at compiled main-text width. Figure 2 reports
  fixed-budget support, corrected axis attribution, and corruption brackets;
  Figure 3 reports the released-oracle/text-inference gap and controlled overflip;
  Figure 4 reports the controlled three-reader paired-exposure effects. The
  observational GateMem evidence and frozen nine-arm retrieval diagnostics remain
  in the appendix. All figures preserve readable labels and uncertainty marks, and
  the selected-settings, nine-arm, matched-prefix, and mechanism-smoke tables remain
  within their text blocks.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
