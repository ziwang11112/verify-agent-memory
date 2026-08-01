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

This repository contains the claim contract, a small pure evaluation library, and
hash-bound normalized aggregate evidence. It contains no benchmark payloads, raw
prompts, model responses, embeddings, checkpoints, reviewer identities, or provider
runtime.

## Current Scope

- Define retrieval admissibility without equating all inadmissible context with
  security harm.
- Keep exposure-to-leakage results associative and reader-specific.
- Separate a small mechanism smoke from the full public-source evaluation.
- Treat released lifecycle and intent fields as an upper bound, not a deployable
  blind inference method.
- Preserve source artifact identities through deterministic normalization.

See [CLAIM_CONTRACT.md](CLAIM_CONTRACT.md), [MIGRATION_ALLOWLIST.md](MIGRATION_ALLOWLIST.md),
and [PROVENANCE.md](PROVENANCE.md) for the frozen boundaries.

## Current Evidence

- Separate reader estimates show a strong non-causal association between target
  exposure and answer leakage.
- Restricting exposure lowers leakage but also reduces bounded utility and increases
  over-refusal.
- Trusted released namespaces improve recall and feasible rate while reducing the
  preregistered penalized conservative risk on the public-source evaluation.
- The evaluated threshold and cluster routers do not establish incremental utility
  beyond namespace support.
- Released intent and lifecycle fields show upper-bound headroom, not deployable
  blind inference performance.
- Two-human agreement is strong on most audited axes, while the prohibited axis is
  prevalence-limited and not reliable enough for a broad claim.

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
```

Regenerating normalized evidence additionally requires the frozen provenance archive
at `../bomi-codex-starter`:

```powershell
uv run --extra dev python scripts/import_frozen_evidence.py
```
