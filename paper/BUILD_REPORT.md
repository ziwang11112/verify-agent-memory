# Paper Build Report

## Receipt

- Built: July 27, 2026.
- Command: `uv run --extra dev --extra paper python -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 14.
- References begin: page 6.
- Appendices begin: page 7.
- NeurIPS checklist begins: page 8.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- All pages were rendered at 120 DPI and visually inspected.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
