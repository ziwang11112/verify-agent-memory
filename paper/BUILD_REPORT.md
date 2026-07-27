# Paper Build Report

## Receipt

- Built: July 27, 2026.
- Command: `uv run --extra dev --extra paper python -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 16.
- Main text ends: page 7.
- References begin: page 8.
- Appendices begin: page 9.
- NeurIPS checklist begins: page 9.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- All pages were rendered at inspection resolution and visually inspected.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- Figures 2 and 3 preserve visible zero references for deltas and remain legible at
  the compiled main-text width.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
