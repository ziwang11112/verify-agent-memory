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
| Query scoring | `src/verify_agent_memory/metrics.py` | Recall, feasibility, non-usable exposure, admissibility violations, bounds, and typed violations |
| Experiment runner | `src/verify_agent_memory/experiment.py` | Setting-by-query execution, source-macro summaries, and dev-only selection |
| Robustness | `src/verify_agent_memory/robustness.py` | Deterministic metadata corruption and observed break-even brackets |
| Pareto analysis | `src/verify_agent_memory/pareto.py` | Fixed-ranking top-k recall, risk, and route-cost frontiers |
| JSON contract | `src/verify_agent_memory/serialization.py` | Strict normalized input and output schemas |
| CLI | `scripts/run_retrieval_experiment.py` | Validation, execution, and setting selection |
| Robustness CLI | `scripts/run_metadata_robustness.py` | Released-to-corrupted metadata curves with fixed retrieval settings |
| Pareto CLI | `scripts/run_top_k_pareto.py` | Exact nested-prefix analysis without reranking or retuning |
| Frozen settings | `experiments/frozen_natural_protocol.json` | Embedding, split, selection, and selected-arm configuration |
| Robustness grid | `experiments/metadata_robustness_protocol.json` | Corruption channels, rates, seeds, invariants, and break-even definition |
| Pareto grid | `experiments/top_k_pareto_protocol.json` | Frozen arms and `top_k` depths for recall-risk-cost analysis |
| Supplemental results | `results/supplemental_natural/` | Content-free top-k, attribution, and metadata break-even diagnostics |
| Supplemental verifier | `scripts/publish_supplemental_results.py` | Hash, schema, row-count, and content-boundary checks for the result package |

No Bayesian mixture, CRP/PYP, split-merge, reader, judge, provider client, or model
call is present in this execution path.

## Normalized Input

The CLI accepts one JSON object per query. A case contains:

- `source` and `group_id`, used for source-macro aggregation and grouped analysis;
- a query ID, namespace, text, frozen embedding, and released intent;
- all method-visible candidate memories, each with a stable ID, namespace, text,
  frozen embedding, released order, and lifecycle state;
- released query-memory policy decisions for content disclosure and operation-trace
  purposes; and
- scorer-only relevance, scope, policy, and lifecycle assessments.

The router never receives scorer assessments. Namespace support is computed from the
released namespace IDs. Policy is not a global `MemoryRecord` property: each case can
supply a `PolicyDecision` for the query-memory pair, and `QueryRecord.policy_purpose`
chooses content-disclosure or operation-trace authorization. This is still a released
metadata approximation, not a general policy engine. The released lifecycle arm can
observe these policy decisions plus released lifecycle and intent fields; ordinary
semantic arms do not use them.

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

The frozen public evaluation contains nine selected arms, all returning at most
`top_k=100` memories. The two `*_only` arms added below are post-hoc attribution
diagnostics with no development tuning; they do not alter the frozen nine-arm run.

| Arm | Candidate support and score |
| --- | --- |
| `global_bm25` | All global candidates; standard BM25 with frozen `k1` and `b` |
| `global_dense` | All global candidates; exact inner product |
| `global_bm25_dense_rrf` | Global BM25 and dense ranks fused as `1/(k+r_bm25) + 1/(k+r_dense)` |
| `global_recency_dense` | Global dense score plus `gamma * normalized_released_order` |
| `namespace_dense` | Exact dense ranking after trusted namespace support restriction |
| `query_agnostic_current_only` | Deliberately misspecified ablation that removes released stale and superseded records for every query |
| `namespace_policy_only` | Namespace support plus query-memory policy filtering, without lifecycle filtering |
| `namespace_lifecycle_only` | Namespace support plus query-intent lifecycle filtering, without policy filtering |
| `released_intent_lifecycle_upper_bound` | Namespace support; policy-disallowed records are removed for every query; current-state queries additionally remove released stale and superseded records, while history queries retain lifecycle states |
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
smallest ranking prefix reaching target recall `0.8`. Relevance and admissibility are
then scored separately on that same utility-matched prefix. Version 2 reports:

- non-usable exposure, which combines relevance and admissibility and is retained for
  compatibility with the frozen v1 experiment;
- admissibility violations alone, with their own known rate, label coverage, and
  lower/upper bounds;
- the four established relevance-by-admissibility cells: relevant/admissible,
  relevant/inadmissible, irrelevant/admissible, and irrelevant/inadmissible;
- joint relevance/admissibility label coverage;
- wrong-scope, policy-disallowed, and lifecycle-incompatible exposure rates;
- route width, candidates scored, and fallback status.

The four cell rates use the whole matched prefix as denominator. Items with unresolved
relevance or admissibility do not enter a known cell, and `joint_label_coverage` makes
that missing mass explicit. The unqualified legacy `contamination_*` Python properties
remain aliases of `non_usable_*`; new JSON output never labels them as pure
admissibility contamination.

An infeasible query receives risk `1.0` only in the penalized development-selection
and aggregate-risk quantity. Conditional matched-prefix metrics remain undefined for
infeasible queries. Queries with no known usable anchors are recall-unevaluable and
are excluded rather than assigned zero.

Every setting summary also decomposes each penalized upper risk into an
`infeasibility_risk_component` and a feasible-prefix conditional-risk contribution.
The identity is computed from query rows within each source before equal-weight
source aggregation. `conditional_*_upper_risk` is reported separately, so a lower
penalized score cannot be described as lower conditional contamination when the gain
actually comes from feasibility.

## Development Selection

Every candidate setting must cover the same source/query rows. The runner first
macro-averages queries within each source and then gives RHELM and MemOps equal
weight. The selection target is explicit in each protocol. It chooses one setting per
arm with this lexicographic objective:

1. maximize feasible rate;
2. minimize the selected penalized upper-bound risk;
3. minimize the corresponding penalized known-label risk;
4. maximize evidence recall;
5. minimize candidates scored; and
6. minimize stable setting ID.

The historical frozen v1 protocol records `selection_risk=non_usable_upper_bound`
because that is what selected the published settings; it is not retroactively renamed
as an admissibility-only result. New selection defaults to
`admissibility_upper_bound`. The selected settings remain immutable evaluation inputs,
and the CLI does not tune or alter them during an evaluation run.

## Metadata Robustness

`robustness.py` changes method-visible metadata while preserving scorer assessments
byte-for-byte. Its deterministic, nested corruption channels cover:

- namespace false allows, false denies, missing labels, and label swaps;
- stale/superseded-to-current errors, current-to-stale errors, and missing lifecycle;
- policy false allows, false denies, and unknown decisions; and
- current/history intent flips.

Namespace swap, missing-label, false-deny, and lifecycle corruption is sampled once
per source-memory identity, so those record fields cannot change merely because
another query references them. Namespace false-allow is instead a query-memory gate
error and is sampled per source-query-memory identity, as are policy decisions;
intent corruption is sampled per source-query. Namespace swaps use one fixed
source-level vocabulary.

For a fixed seed and channel, cases corrupted at rate `r1` are a subset of those
corrupted at any larger rate `r2`. Retrieval settings never change along a curve. The
break-even utility reports the last contiguous observed rate where namespace dense
weakly dominates global dense on both feasible rate and penalized admissibility upper
risk, followed by the first observed non-dominating rate. It is a grid bracket, not an
interpolated population threshold.

The lifecycle rule remains a two-intent released-field approximation. It does not
claim that every stale record is valid for every historical question, and corruption
experiments do not turn it into temporal inference.

## Top-k Pareto Analysis

The Pareto runner produces each frozen ranking once at the maximum requested depth
and rescoring uses exact nested prefixes at `top_k` 5, 10, 20, 50, and 100. It reports
feasibility, non-usable and admissibility risk, returned memories, candidates scored,
and route width under identical ranking parameters. Candidate counts and returned
memories are algorithmic work measures. Local wall-clock observations are not eligible
for a deployment-latency claim.

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
uv run --extra dev python -m scripts.run_metadata_robustness `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --retrieval-protocol experiments/frozen_natural_protocol.json `
  --robustness-protocol experiments/metadata_robustness_protocol.json `
  --output tmp/metadata_curve.jsonl `
  --break-even-output tmp/metadata_break_even.jsonl
uv run --extra dev python -m scripts.run_top_k_pareto `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --retrieval-protocol experiments/frozen_natural_protocol.json `
  --pareto-protocol experiments/top_k_pareto_protocol.json `
  --output tmp/top_k_pareto.jsonl
uv run --extra dev python -m scripts.publish_supplemental_results verify
```

The fixture is only a code-path smoke. It is not a benchmark result. Reproducing the
frozen natural-corpus run requires separately acquired public-source data and the
hash-bound source construction described in `PROVENANCE.md`.
