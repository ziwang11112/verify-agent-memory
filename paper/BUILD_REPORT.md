# Paper Build Report

## Receipt

- Built: July 28, 2026.
- Command: `uv run --extra dev --extra paper python -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 19.
- Main text ends: page 9.
- References begin: page 9.
- Appendices begin: page 10.
- NeurIPS checklist begins: page 12.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- All pages were rendered at inspection resolution and visually inspected.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- Figure 1's four audited query-memory cases, use/drop decisions, and compact
  verification trace remain legible at compiled main-text width. Figures 2 and 3
  preserve visible zero references for deltas, direct method labels, and readable
  uncertainty marks. The nine-arm and matched-prefix tables remain within their
  text blocks.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
