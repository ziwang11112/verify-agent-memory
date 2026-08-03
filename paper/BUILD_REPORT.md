# Paper Build Report

## Receipt

- Built: August 3, 2026.
- Command: `.venv\Scripts\python.exe -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 22.
- Main text ends: page 9.
- References begin: page 10.
- Appendices begin: page 11.
- NeurIPS checklist begins: page 16.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- All 22 pages were rendered at inspection resolution and visually inspected,
  including methodology pages 3--4, the expanded experimental setup and metric
  equations on pages 4--7, results through page 9, the main-text/reference transition
  across pages 9--10, appendix figures and tables through page 16, and the checklist.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- Section 4 now exposes the datasets and splits, retrieval and reader model snapshots,
  complete comparison families, formula-level definitions of retrieval, calibration,
  filter, controlled-verification, and exposure metrics, dev-only selection,
  uncertainty procedure, and interpretation boundaries in the main text. Table 1 is
  a three-column experiment/data--systems--outcomes map; full hashes, derivations,
  edge-case conventions, and selected hyperparameters remain in the appendix for
  auditability.
- Figure 1 links an audited candidate-level retrieval failure, a query-conditioned
  prompt boundary, and the three-stage evidence roadmap. Figure 2 presents the value, source, and
  fragility of metadata constraints in a single mechanism sequence. Figure 3 shows
  the released-oracle/text-inference gap and controlled overflip; Figure 4 shows the
  controlled three-reader paired-exposure effects and residual leakage. The four
  additional audited examples, observational GateMem evidence, and frozen nine-arm
  retrieval diagnostics remain in the appendix. All figures preserve readable
  labels and uncertainty marks, and all generated tables remain within their text
  blocks.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- The flat, single-file Overleaf export compiled successfully with the same assets.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
