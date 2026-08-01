# Paper Package

This package targets the NeurIPS 2026 workshop **Who Verifies the Agents? Toward
Reliable Agent Development**. The submission uses the official double-blind workshop
mode and a 4-9 content-page limit.

## Build Evidence Artifacts

```powershell
uv sync --extra dev --extra paper
uv run --extra dev --extra paper python -m scripts.build_paper_artifacts
uv run --extra dev --extra paper python -m scripts.verify_paper
```

The generator reads only `evidence/normalized/*.csv` after
`scripts/verify_evidence.py` succeeds. It writes all empirical LaTeX numbers, tables,
and figures under `paper/generated/`, plus a hash manifest binding inputs, outputs,
and generator code. Figure assets are emitted as vector PDF/SVG and inspection PNG:

- `verification_pipeline.*`: four abridged audited examples covering wrong
  namespace, supersession, explicit forgetting, and history-compatible old states,
  plus the compact record-level verification interface;
- `evidence_summary.*`: exposure/disclosure, reader calibration, filtering
  trade-off;
- `retrieval_results.*`: all retrieval arms in recall--cost space, the trusted
  namespace effect, incomplete-label uncertainty, and router/lifecycle contrasts.

`paper/FIGURE_CONTRACT.md` records each figure's intended conclusion and claim
boundary.

## Compile

With `pdflatex` and BibTeX available on `PATH`:

```powershell
uv run python -m scripts.compile_paper
```

The script does not require `latexmk` or Perl. The compiled PDF is intentionally
ignored by Git. Generated source tables and figure assets are tracked. The latest
page-count and visual-inspection receipt is in `paper/BUILD_REPORT.md`.

## Review Gates

- `paper/main.tex` must use `dblblindworkshop`.
- `paper/checklist.tex` must contain no TODO answers.
- C1-C7 must each appear in `paper/CLAIM_MAP.md`.
- Generated artifact hashes must match the manifest.
- No empirical number should be hand-entered in prose when a generated macro exists.
- The main text must not use retired internal method labels.
- The paper must remain explicit that the source evaluation is not an official
  leaderboard submission.
