# The Wrong Memory at the Right Time

**Reproducible evaluation artifact for _Verifying Retrieval Admissibility in
Long-Term Agents_.**

Long-term-memory agents can retrieve a record that is topically relevant but still
ineligible for the current principal, policy, intent, lifecycle state, or time. This
repository provides the evaluation code, frozen protocols, derived evidence, and
integrity checks used to trace that failure across four observable stages:

```text
Stored -> Retrieved -> Exposed -> Disclosed
```

The package name remains `verify-agent-memory` for stable imports and commands. This
is an evaluation and verification artifact, not a new memory index, a production
policy engine, or an official benchmark leaderboard.

## What Is Verified

| Stage | Verification question | Main outputs |
| --- | --- | --- |
| Candidate support | Was the right evidence reachable under the allowed namespace? | Evidence recall, target-recall feasibility, candidate work |
| Eligibility | Was each retrieved record allowed for this query and time? | Three-valued decisions, typed violations, coverage, risk bounds |
| Exposure | Which record IDs crossed the prompt boundary? | Identity-preserving prompt traces |
| Disclosure | Did exposed content appear in the answer? | Reader-specific answer and literal-disclosure effects |

For memory `m` and query context `z = (q, principal, policy, intent, time)`:

```text
admissible_z(m) = scope_z(m) AND policy_z(m) AND lifecycle_z(m)
usable_z(m)     = relevant_z(m) AND admissible_z(m)
```

Every component is three-valued: positive, negative, or unresolved. Known violations
are excluded, while missing evidence remains unresolved and is reflected in coverage
and lower/upper risk bounds. Routes are compared at matched evidence recall so that
returning less useful context cannot appear artificially safe.

See [`docs/EXPERIMENT_METHODS.md`](docs/EXPERIMENT_METHODS.md) for the executable
method contract and [`CLAIM_CONTRACT.md`](CLAIM_CONTRACT.md) for reporting limits.

## Results at a Glance

The checked-in natural evaluation contains 87 namespace groups, 182,908 memories,
and 3,767 RHELM/MemOps queries. On frozen top-20 rankings, trusted namespace support:

- raises evidence recall from `0.432` to `0.533`;
- raises the fraction reaching the `0.8` recall target from `0.237` to `0.311`;
- reduces exact candidate scoring from 90,122 to 1,551 candidates (`58.1x` fewer);
- reduces feasible prefixes containing any known admissibility violation from
  `0.528` to `0.396`; and
- reduces the mean known-violation count from `2.080` to `1.393`, without reaching
  zero.

Correct provenance identity is necessary for this result: a size-matched random
partition performs poorly, and global retrieval requires depth 500 before late
namespace filtering approaches namespace pre-filter recall. The namespace result is
therefore a benchmark-conditional candidate-support intervention, not a safety
certificate or a demonstrated policy/lifecycle solution.

On a frozen 1,523-case route-to-reader subset, namespace-minus-global answer-accuracy
effects are positive for separately reported DeepSeek, Gemini, and sequential GPT
readers (`+0.053` to `+0.068`). These route contrasts are observational and use one
shared blinded primary judge.

The remaining diagnostics establish important boundaries:

- on 72 public-development cases, a released-field oracle lowers
  recall-constrained upper loss by `0.032`, while the two tested text-only gates do
  not realize that headroom;
- all four controlled readers show positive selectivity between relevant-admissible
  and relevant-inadmissible evidence, but DeepSeek retains a `+0.156`
  relevant-inadmissible disclosure effect; and
- protected- and stale-disclosure changes are inconclusive, so the artifact makes no
  general disclosure-reduction claim.

The top-100 v1 route-family evaluation is confirmatory for its historical
non-usable metric. The top-20 admissibility result above is a post-hoc v2 rescore of
the same frozen rankings: it changes no route, ranking, setting, or hyperparameter.
Exact estimates, intervals, and provenance are indexed in
[`results/README.md`](results/README.md), [`evidence/README.md`](evidence/README.md),
and [`claims/claims.yaml`](claims/claims.yaml).

## Quick Start

Requirements: Python 3.11+, Git, and [`uv`](https://docs.astral.sh/uv/). The default
path requires no API key, provider call, GPU, or private data.

```powershell
uv sync --extra dev --extra plots

uv run --extra dev python -m scripts.run_retrieval_experiment validate `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json

uv run --extra dev python -m scripts.run_retrieval_experiment run `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/retrieval_routes.jsonl
```

These synthetic fixtures exercise the same normalized interfaces and invariants as
the natural evaluation. They are implementation checks, not benchmark estimates.

## Verify the Released Artifact

Run the complete offline verification path:

```powershell
uv run --extra dev --extra plots python -m pytest
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
```

Verify each derived result family and regenerate released plots with the commands in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The checks reject schema drift,
non-finite values, changed hashes, incomplete result families, release-excluded
paths, and common secret patterns.

## Data and Reproducibility Boundary

| Material | Availability |
| --- | --- |
| Evaluation library, protocols, prompts, and synthetic fixtures | Included |
| Content-free aggregate, pair-level, and tokenized derived scores | Included and hash-bound |
| GateMem, RHELM, and MemOps source repositories | Fetchable at pinned commits and tree hashes |
| Raw benchmark text, provider requests/responses, embeddings, and credentials | Excluded |

Fetch and verify redistributable public sources from their owners:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources validate
uv run --extra dev python -m scripts.fetch_public_sources fetch
uv run --extra dev python -m scripts.fetch_public_sources verify
```

The public derivative records reconstruct the released aggregates, source-specific
contrasts, bootstrap intervals, and figures without a provider call. They cannot
independently audit the original private payload-to-provider-to-score
transformation. See [`data/README.md`](data/README.md),
[`PROVENANCE.md`](PROVENANCE.md), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Repository Map

```text
src/           pure evaluation library
scripts/       experiment runners, publishers, and integrity checks
experiments/   frozen protocols, prompts, and constructed inputs
data/          public-source registry and redistribution policy
results/       content-free derived result packages
evidence/      normalized measurements and provenance manifests
claims/        machine-readable interpretation boundaries
tests/         unit, invariant, provenance, and reproducibility tests
docs/          method and execution contracts
```

Original code and documentation are MIT licensed. Third-party datasets and source
repositories retain their own terms. Citation metadata will be added after the
double-blind review period; during review, refer to the accompanying paper by title.
