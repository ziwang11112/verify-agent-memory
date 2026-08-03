# The Wrong Memory at the Right Time

**Verifying Retrieval Admissibility in Long-Term Agents**

Long-term memory is intended to prevent language agents from forgetting, but
persistence creates a complementary verification problem: a retrieved memory can
be semantically relevant while being out of scope, prohibited, temporally
incompatible, or otherwise inappropriate for the current question. **A memory can
be relevant and still be inadmissible for the current query.** This project studies
how to verify that distinction while preserving the useful evidence an agent needs.

The evaluation surface separates four states:

```text
Stored -> Retrieved -> Exposed -> Disclosed
```

Retrieval does not imply prompt exposure, and exposure does not imply answer
disclosure.
The formal object is query-conditioned admissibility:

```text
admissible(m, q, p, t)
  = scopeAllowed(m, q, p)
    AND policyAllowed(m, q, p)
    AND lifecycleCompatible(m, intent(q), t)

usable(m, q, p, t)
  = relevant(m, q)
    AND admissible(m, q, p, t)
```

This repository contains the claim contract, a small pure evaluation library,
hash-bound controlled prompts, and normalized aggregate evidence. It contains no
private benchmark payloads, raw provider responses, credentials, embeddings, or
checkpoints. Provider-backed historical diagnostics remain isolated from the public
retrieval evaluator.

For the code-level experiment contract, all nine non-Bayesian retrieval arms, the
dev-only setting-selection objective, and local smoke commands, see
[docs/EXPERIMENT_METHODS.md](docs/EXPERIMENT_METHODS.md). This technical path is
independent of the manuscript build.

The controlled counterfactual diagnostic holds each candidate pool fixed while
changing only principal, intent, purpose, or as-of time. Its frozen protocol, pure
scorer, and content-free results are documented in
[docs/COUNTERFACTUAL_ADMISSIBILITY_EXPERIMENT.md](docs/COUNTERFACTUAL_ADMISSIBILITY_EXPERIMENT.md)
and [results/counterfactual_admissibility/](results/counterfactual_admissibility/).

The paired exposure intervention holds the query and two background candidates fixed
while including or withholding one candidate. Its purpose-built 2x2 candidate
contract, literal disclosure scorer, and zero-call construction commands are documented in
[docs/PAIRED_EXPOSURE_INTERVENTION.md](docs/PAIRED_EXPOSURE_INTERVENTION.md) and
[experiments/counterfactual_exposure_protocol.json](experiments/counterfactual_exposure_protocol.json).
That construction CLI still has no provider client or execution command. A separately
authorized, hash-bound execution completed the frozen panel for GPT-5.6 Sol, Gemini
3.6 Flash, and DeepSeek V4 Pro. Only content-free derived scores, aggregates, costs,
and receipts are published in
[results/counterfactual_exposure/](results/counterfactual_exposure/); raw prompts and
responses remain outside this repository.

## Current Scope

- Define retrieval admissibility without equating all inadmissible context with
  security harm.
- Keep natural-route exposure-to-leakage results associative, and bound the separate
  paired intervention to controlled prompt-level disclosure effects.
- Separate a small mechanism smoke from the full public-source evaluation.
- Keep the frozen historical released-field v1 arm distinct from the corrected v2
  governance diagnostic; neither is a deployable blind inference method.
- Preserve source artifact identities through deterministic normalization.

See [CLAIM_CONTRACT.md](CLAIM_CONTRACT.md), [MIGRATION_ALLOWLIST.md](MIGRATION_ALLOWLIST.md),
and [PROVENANCE.md](PROVENANCE.md) for the frozen boundaries.

## Current Evidence

- Separate reader estimates show a strong non-causal association between target
  exposure and answer leakage.
- Restricting exposure lowers leakage but also reduces bounded utility and increases
  over-refusal.
- Trusted released namespaces improve recall and feasible rate while reducing the
  preregistered penalized non-usable upper risk on the public-source evaluation.
- The evaluated threshold and cluster routers do not establish incremental utility
  beyond namespace support.
- Corrected v2 attribution shows that released policy metadata drives the incremental
  governance gain; the evaluated coarse lifecycle-only rule hurts retrieval.
- In controlled counterfactual pairs, GPT-5.6 Sol and Gemini 3.6 Flash follow every
  focal eligibility flip, but all three complete providers fail the preregistered
  stable-control overflip ceiling. Explicit LLM verification is therefore not yet a
  selective replacement for trusted controls.
- In the paired exposure intervention, the relevant-admissible minus
  relevant-inadmissible disclosure effect is 0.906 for GPT-5.6 Sol, 0.812 for Gemini
  3.6 Flash, and 0.656 for DeepSeek V4 Pro; all scenario-bootstrap intervals exclude
  zero. DeepSeek also has a positive relevant-inadmissible exposure effect of 0.156,
  so reader selectivity is not an enforcement boundary.

The normalized values and their source receipts are documented in
[evidence/README.md](evidence/README.md).

## Paper

The double-blind NeurIPS 2026 workshop manuscript, claim map, references, and
official checklist are under [paper/](paper/). Every empirical table, figure, and
prose macro is regenerated from verified normalized evidence:

```powershell
uv sync --extra dev --extra paper
uv run --extra dev --extra paper python -m scripts.build_paper_artifacts
uv run --extra dev --extra paper python -m scripts.verify_paper
uv run --extra dev --extra paper python -m scripts.compile_paper
```

The compile step requires `pdflatex` and BibTeX on `PATH`. The submission target
and page-limit receipt are recorded in
[paper/SUBMISSION_TARGET.md](paper/SUBMISSION_TARGET.md).

## Verify

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.publish_counterfactual_exposure_results verify
```

Regenerating normalized evidence additionally requires the frozen provenance archive
at `../bomi-codex-starter`:

```powershell
uv run --extra dev python scripts/import_frozen_evidence.py
```
