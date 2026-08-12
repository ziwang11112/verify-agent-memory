# Text-Inferred Admissibility Diagnostic

This content-free package reports a public-development diagnostic of the gap between
released governance fields and text-only inference. It is not an official RHELM or
MemOps benchmark result. The frozen sample has 96 queries: 24 calibration cases and
72 analysis cases, equally divided among RHELM current-state, MemOps current-state,
and MemOps history strata. Every route begins from the same namespace-local top-20
ranking.

## Main Result

Released policy and lifecycle fields provide measurable headroom. Compared with
namespace dense, the released-field oracle preserves exactly the same evidence recall
and feasible rate while reducing equal-stratum-macro penalized admissibility risk by
`0.0320` (95% group-bootstrap CI `[-0.0651, -0.0067]`) and removing 3.06 candidates per
route on average.

Neither text-only model realizes that headroom. Both inferred filters use a violation
threshold of `0.95`, selected on calibration with zero required-anchor false denials.
On the held-aside analysis portion, however, both filters lose recall and increase the
primary penalized risk.

| Provider / arm | Evidence recall | Feasible rate | Penalized upper risk | Mean route width |
| --- | ---: | ---: | ---: | ---: |
| Namespace dense | 0.5252 | 0.2778 | 0.7553 | 20.00 |
| Released-field oracle | 0.5252 | 0.2778 | 0.7234 | 16.94 |
| GPT-5.6 text inferred | 0.4836 | 0.1944 | 0.8121 | 18.07 |
| Gemini 3.6 text inferred | 0.5113 | 0.2500 | 0.7733 | 19.60 |

Paired deltas below are inferred minus namespace dense. Negative recall and feasible
rate are worse; positive risk is worse.

| Provider | Recall delta (95% CI) | Feasible-rate delta (95% CI) | Risk delta (95% CI) |
| --- | ---: | ---: | ---: |
| GPT-5.6 | -0.0417 [-0.0667, -0.0167] | -0.0833 [-0.1333, -0.0333] | +0.0567 [+0.0238, +0.0916] |
| Gemini 3.6 | -0.0139 [-0.0333, 0.0000] | -0.0278 [-0.0667, 0.0000] | +0.0180 [0.0000, +0.0432] |

The abstaining verifier selects an unknown cutoff of `0.2`, but its filtering and
routing outputs are identical to the direct text-inferred arm for both providers.
Abstention therefore adds no downstream value in this frozen grid.

## Why Inference Fails

GPT-5.6 contains weak-to-moderate admissibility discrimination (equal-stratum-macro
ROC-AUC `0.6262`, PR-AUC `0.3515`) but does not separate violations from required
evidence reliably enough to filter. Its analysis violation recall is `0.2320` at
precision `0.3502`, and it falsely denies 6 of 113 required anchors. The resulting
false-deny rate is `0.0588`, despite zero false denials on calibration.

Gemini is more conservative. Its violation precision is `0.6786`, but recall is only
`0.0486`; it falsely denies 2 of 113 required anchors. Its admissibility ROC-AUC is
`0.5222`, close to chance. Most failures concentrate in MemOps current-state queries,
where GPT-5.6 denies 6 of 34 anchors and Gemini denies 2 of 34.

The observed result is therefore not that governance verification is useless. The
released-field oracle improves risk without recall loss, with a CI excluding zero.
The result is that visible text, time, and order alone do not identify those governance
facts reliably enough for hard filtering in this sample. Trusted provenance fields
and explicit lifecycle/policy state remain materially different from semantic model
judgment.

## Provider Completeness

OpenAI and Gemini complete all 96 frozen cases and are comparison-eligible. DeepSeek
returns 60 valid cases before one strict-parser `ValueError`; its prefix is retained for
execution audit but is neither scored nor rerun. Claude Sonnet 5 passes the synthetic
20-candidate fixture but returns fewer than 20 candidates on the first public-dev case;
it is also not rerun or scored. These are structured-output contract failures, not
quality comparisons.

For complete primary checkpoints, the configured price estimate is `$7.4432` for
GPT-5.6 and `$2.2408` for Gemini. Mean per-case latency is 20.56 s and 8.92 s,
respectively; p95 latency is 25.84 s and 13.94 s. These are client-observed latencies
and protocol-price estimates, not production throughput claims.

## Recovery Sensitivity

The Gemini raw checkpoint contains seven duplicate calls caused by overlapping resume
processes. The primary rule takes the first response committed for each bound request;
the last committed response is a separate sensitivity view. Two duplicate groups have
different probability outputs, but selected thresholds, filter metrics, route metrics,
and every paired bootstrap delta are byte-identical across the two views. Raw files are
preserved and the extra `$0.1337` configured cost is reported.

## Boundaries

- This is a balanced 96-query development diagnostic, not a population estimate.
- Thresholds use calibration cases only; the 72 analysis cases are not retuned.
- Models see no namespace IDs, source IDs, released labels, required-evidence labels,
  embeddings, or scorer metadata.
- The oracle uses released policy/lifecycle fields and is not a deployable blind method.
- Non-anchor topical relevance is unresolved in these sources, so the package reports
  admissibility violations rather than claiming a full relevant-and-inadmissible rate.
- No answer generation, reader, judge, held-out access, or official benchmark scoring
  occurs in this diagnostic.

Run `python -m scripts.publish_inferred_admissibility_results verify` to verify every
published file and provenance receipt without accessing local provider responses.
