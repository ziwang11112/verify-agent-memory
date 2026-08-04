# Paper Build Report

## Receipt

- Built: August 4, 2026.
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
- Section 4 now exposes the datasets and splits, retrieval and reader model
  identifiers and execution controls, visible and hidden verifier fields, calibration
  grids and tie-breaks, complete comparison families, formula-level definitions,
  dev-only selection, exact bootstrap units, and interpretation boundaries in the
  main text. Table 1 is
  a three-column experiment/data--systems--outcomes map; full hashes, derivations,
  edge-case conventions, and selected hyperparameters remain in the appendix for
  auditability.
- Figure 1 is a compact four-row query--candidate audit matrix with explicit scope,
  lifecycle, and policy decisions plus three visibly separate empirical populations.
  Figure 2 presents the value, source, and channel-specific fragility of metadata
  constraints, including the distinct global-dense and clean-namespace references.
  Figure 3 shows the released-oracle/text-inference gap and controlled overflip;
  Figure 4 shows reader-specific literal-marker exposure effects and the separately
  executed Claude Opus 5 replication without pooling. The four additional audited
  examples, observational GateMem evidence, and frozen nine-arm retrieval diagnostics
  remain in the appendix. All figures preserve readable labels and uncertainty marks,
  and all generated tables remain within their text blocks.
- The main results report the 58.1-fold reduction in mean candidates scored for clean
  namespace support and explicitly bound it as exact-search retrieval work, not
  wall-clock or production latency. Related work now includes ABAC, Securing the
  Agent, SD-RAG, Collaborative Memory, and CIMemories while distinguishing prior
  enforcement mechanisms from the paper's stage-specific evaluation contribution.
- The reproducibility section and checklist now state the public-artifact boundary
  precisely: released derived scores rebuild every table and figure, but separately
  retained prompt/response bundles are required to audit provider-output-to-score
  transformations or rerun provider executions.
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
