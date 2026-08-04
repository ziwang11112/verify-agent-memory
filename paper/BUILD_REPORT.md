# Paper Build Report

## Receipt

- Built: August 3, 2026.
- Command: `.venv\Scripts\python.exe -m scripts.compile_paper`.
- Engine: MiKTeX pdfTeX 1.40.28 with BibTeX 0.99e.
- Output: `paper/build/main.pdf` (ignored by Git).
- Total pages: 23.
- Main text ends: page 9.
- References begin: page 9, after the conclusion.
- Appendices begin: page 10.
- NeurIPS checklist begins: page 16.
- Content length: within the workshop's 4-9 page limit, excluding references and
  appendices.

## Validation

- All 23 pages were rendered at inspection resolution and visually inspected,
  including methodology pages 3--4, experimental setup and the central risk equation
  on pages 4--6, results through page 9, the main-text/reference transition on page 9,
  relocated diagnostic equations from page 10 onward, appendix figures and tables,
  and the checklist from page 16.
- No clipped text, overlapping elements, broken tables, or unreadable figures were
  observed.
- Section 4 now exposes the datasets and splits, retrieval and reader model snapshots,
  complete comparison families, formula-level definitions of retrieval, calibration,
  filter, controlled-verification, and exposure metrics, dev-only selection,
  uncertainty procedure, and interpretation boundaries in the main text. Table 1 is
  a three-column experiment/data--systems--outcomes map; full hashes, derivations,
  edge-case conventions, and selected hyperparameters remain in the appendix for
  auditability.
- Figure 1 links three audited candidate-level failures (wrong scope, superseded
  current state, and an explicit forget target) to the query-conditioned prompt
  boundary and the three-stage evidence roadmap. Figure 2 presents the value, source,
  and fragility of metadata constraints in a single mechanism sequence. Figure 3 shows
  the released-oracle/text-inference gap and controlled overflip; Figure 4 shows the
  original three-reader paired-exposure effects, residual leakage, and the separately
  executed Claude Opus 5 replication without pooling. The four additional audited
  examples, observational GateMem evidence, and frozen nine-arm retrieval diagnostics
  remain in the appendix. All figures preserve readable labels and uncertainty marks,
  and all generated tables remain within their text blocks.
- The main results report the 58.1-fold reduction in mean candidates scored for clean
  namespace support and explicitly bound it as exact-search retrieval work, not
  wall-clock or production latency. Related work now includes Collaborative Memory's
  governance architecture and CIMemories' prompt-level all-or-nothing precedent,
  while distinguishing the paper's record-level estimands.
- The official double-blind workshop style injects `Anonymous Author(s)`,
  `Affiliation`, `Address`, and `email` on page 1. These strings are template output,
  not stale manuscript metadata; the official style file is unchanged.
- The final log contains no overfull box, undefined citation, undefined reference,
  rerun, or LaTeX error warning.
- The flat, single-file Overleaf export compiled successfully with the same assets.
- Remaining underfull box messages are benign line-breaking notices in body text and
  long bibliography URLs.
- Generated TeX, PDF, PNG, and manifest hashes were unchanged across two consecutive
  artifact builds.
