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
Stored -> Retrieved -> Exposed -> Used
```

Retrieval does not imply prompt exposure, and exposure does not imply answer use.
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

This repository is initially a claim-governance skeleton. It records the evidence
contract, source provenance, migration boundary, and automated checks that must pass
before any implementation or frozen aggregate is imported. It currently contains no
benchmark payloads, model responses, embeddings, checkpoints, or provider runtime.

## Current Scope

- Define retrieval admissibility without equating all inadmissible context with
  security harm.
- Keep exposure-to-leakage results associative and reader-specific.
- Separate a small mechanism smoke from the full public-source evaluation.
- Treat released lifecycle and intent fields as an upper bound, not a deployable
  blind inference method.
- Preserve source artifact identities before any deterministic normalization.

See [CLAIM_CONTRACT.md](CLAIM_CONTRACT.md), [MIGRATION_ALLOWLIST.md](MIGRATION_ALLOWLIST.md),
and [PROVENANCE.md](PROVENANCE.md) for the frozen boundaries.
