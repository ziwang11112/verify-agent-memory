# Retrieval Experiment Code Guide

This document is an implementation map, not a manuscript or a result summary. It
describes the executable non-Bayesian retrieval-admissibility experiment exposed by
this repository.

## Code Map

| Layer | Entry point | Responsibility |
| --- | --- | --- |
| Typed labels | `src/verify_agent_memory/schema.py` | Relevance, scope, lifecycle, query intent, and scorer-only assessments |
| Admissibility | `src/verify_agent_memory/admissibility.py` | Three-valued admissibility and usability decisions |
| Retrieval arms | `src/verify_agent_memory/retrieval.py` | Candidate support, ranking, filtering, clustering, and fallback |
| Query scoring | `src/verify_agent_memory/metrics.py` | Recall, feasibility, matched-prefix contamination, bounds, and typed violations |
| Experiment runner | `src/verify_agent_memory/experiment.py` | Setting-by-query execution, source-macro summaries, and dev-only selection |
| JSON contract | `src/verify_agent_memory/serialization.py` | Strict normalized input and output schemas |
| CLI | `scripts/run_retrieval_experiment.py` | Validation, execution, and setting selection |
| Frozen settings | `experiments/frozen_natural_protocol.json` | Embedding, split, selection, and selected-arm configuration |

No Bayesian mixture, CRP/PYP, split-merge, reader, judge, provider client, or model
call is present in this execution path.

## Normalized Input

The CLI accepts one JSON object per query. A case contains:

- `source` and `group_id`, used for source-macro aggregation and grouped analysis;
- a query ID, namespace, text, frozen embedding, and released intent;
- all method-visible candidate memories, each with a stable ID, namespace, text,
  frozen embedding, released order, lifecycle state, and optional policy field; and
- scorer-only relevance and scope assessments.

The router receives `MemoryRecord` and `QueryRecord`. It never receives relevance or
scope assessment labels. Namespace support is computed from the released namespace
IDs. The released lifecycle upper-bound arm can observe the explicitly supplied
lifecycle, intent, and policy fields; other semantic arms do not use those fields.

Raw RHELM, MemOps, or GateMem payloads are not distributed here. The synthetic file
`tests/fixtures/retrieval_cases.jsonl` exercises the same normalized interface without
private data or benchmark redistribution.

## Candidate Construction

The frozen protocol records the source-specific construction rules:

1. RHELM groups are stable persona hashes. A query can see conversation turns in its
   split that occur strictly before the query date. Namespace arms then restrict this
   global support to the query persona.
2. MemOps groups are stable profile hashes. Every method-visible dialogue memory in
   the split enters global support. Namespace arms restrict support to the query
   profile. Released source order supplies the recency diagnostic because profiles do
   not share a cross-case wall clock.
3. There is no candidate subsampling and no positive injection.
4. Development and evaluation groups are disjoint. Evaluation settings are frozen
   before evaluation rows are processed.

## Embedding and Ranking Contract

The checked-in protocol identifies Qwen3-Embedding-8B and its exact revision. Inputs
use 4,096-dimensional float32 embeddings, a 512-token maximum, normalized last-token
pooling, and unit L2 normalization. Dense ranking is exact inner product. Equal scores
are resolved by ascending memory ID.

The public runner consumes frozen vectors rather than downloading or executing the
embedding model. This keeps retrieval deterministic and keeps model/provider access
outside the evaluator.

## Retrieval Arms

All selected arms return at most `top_k=100` memories.

| Arm | Candidate support and score |
| --- | --- |
| `global_bm25` | All global candidates; standard BM25 with frozen `k1` and `b` |
| `global_dense` | All global candidates; exact inner product |
| `global_bm25_dense_rrf` | Global BM25 and dense ranks fused as `1/(k+r_bm25) + 1/(k+r_dense)` |
| `global_recency_dense` | Global dense score plus `gamma * normalized_released_order` |
| `namespace_dense` | Exact dense ranking after trusted namespace support restriction |
| `namespace_current_only` | Namespace support with released stale and superseded records removed for every query |
| `released_intent_lifecycle_upper_bound` | Namespace support; current-state queries remove released stale, superseded, and policy-disallowed records; history queries retain lifecycle states |
| `threshold_router` | Online namespace-local centroid assignment at cosine threshold `theta`; route to at most `top_l` qualifying clusters |
| `cluster_router` | Online namespace-local A5 centroid assignment and routing with size, cosine, and new-cluster scores |

The threshold router assigns a memory to the most similar existing centroid when its
cosine is at least `theta`; otherwise it creates a cluster. At query time it selects
at most `top_l` centroids meeting the same threshold.

For the non-Bayesian cluster router, the existing- and new-cluster insertion scores
are implemented as:

```text
existing(k, x) = gamma * log(n_k + beta) + cosine(x, centroid_k) / tau
new(x)         = log(alpha_like) + (1 - max_cosine(x, centroid)) / tau_new
```

Query routing ranks existing clusters by the existing-cluster score and selects the
best `top_l`. Both clustered arms rank selected memories by exact dense similarity.
An empty clustered route falls back to namespace-local dense support, never global
support.

## Scoring

For each query, known usable memories are recall anchors. The scorer finds the
smallest ranking prefix reaching target recall `0.8`. On that prefix it reports:

- known contamination among resolved labels;
- resolved-label coverage;
- lower and upper contamination bounds obtained by assigning every unresolved item
  to usable and contaminating, respectively;
- wrong-scope, policy-disallowed, and lifecycle-incompatible exposure rates;
- route width, candidates scored, and fallback status.

An infeasible query receives risk `1.0` only in the penalized development-selection
and aggregate-risk quantity. Conditional matched-prefix metrics remain undefined for
infeasible queries. Queries with no known usable anchors are recall-unevaluable and
are excluded rather than assigned zero.

## Development Selection

Every candidate setting must cover the same source/query rows. The runner first
macro-averages queries within each source and then gives RHELM and MemOps equal
weight. It chooses one setting per arm with this lexicographic objective:

1. maximize feasible rate;
2. minimize penalized conservative upper-bound risk;
3. minimize penalized resolved contamination;
4. maximize evidence recall;
5. minimize candidates scored; and
6. minimize stable setting ID.

The selected settings in `experiments/frozen_natural_protocol.json` are immutable
evaluation inputs. The CLI does not tune or alter them during an evaluation run.

## Local Smoke

```powershell
uv sync --extra dev
uv run --extra dev python -m scripts.run_retrieval_experiment validate `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json
uv run --extra dev python -m scripts.run_retrieval_experiment run `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/retrieval_routes.jsonl
uv run --extra dev python -m scripts.run_retrieval_experiment select `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/selected_settings.jsonl
```

The fixture is only a code-path smoke. It is not a benchmark result. Reproducing the
frozen natural-corpus run requires separately acquired public-source data and the
hash-bound source construction described in `PROVENANCE.md`.
