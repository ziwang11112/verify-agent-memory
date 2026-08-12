# Supplemental Natural-Corpus Diagnostics

This package contains post-hoc, content-free diagnostics over the frozen natural
evaluation population: 87 groups, 182,908 memories, and 3,767 queries from the
public RHELM and MemOps sources. It is not an official benchmark submission or a
replacement for the frozen primary evaluation. No retrieval setting was retuned,
and no provider, reader, judge, or paid call was made.

## Fixed-Budget Frontier

The top-k analysis reuses each frozen top-100 ranking and scores exact nested
prefixes. The namespace effect is present at practical depths, rather than appearing
only at the development-selected `top_k=100`.

| Arm | top-k | Evidence recall | Feasible rate | Penalized admissibility upper risk |
| --- | ---: | ---: | ---: | ---: |
| Global dense | 10 | 0.3223 | 0.1703 | 0.8461 |
| Namespace dense | 10 | 0.3895 | 0.2021 | 0.8074 |
| Global dense | 20 | 0.4316 | 0.2374 | 0.7981 |
| Namespace dense | 20 | 0.5329 | 0.3113 | 0.7127 |
| Global dense | 50 | 0.6032 | 0.3991 | 0.6984 |
| Namespace dense | 50 | 0.7399 | 0.5517 | 0.5185 |
| Global dense | 100 | 0.7168 | 0.5389 | 0.6276 |
| Namespace dense | 100 | 0.8657 | 0.7619 | 0.3537 |

At `top_k=100`, source-macro candidates scored fall from 90,121.7 for global
dense to 1,551.2 for namespace dense, a 98.28% algorithmic-work reduction. This is
not a production-latency claim. Threshold routing is below namespace dense throughout
the measured frontier; cluster routing nearly aliases it.

## Axis Attribution

The attribution rerun uses the same frozen dense encoder and exact ranking rule. It
adds no development selection. Deltas below are method minus namespace dense with a
10,000-sample namespace-group bootstrap.

| Filter | Recall delta | Feasible-rate delta | Penalized-risk delta |
| --- | ---: | ---: | ---: |
| Policy only | +0.0265 [0.0223, 0.0308] | +0.0576 [0.0482, 0.0675] | -0.1467 [-0.1548, -0.1388] |
| Lifecycle only | -0.0158 [-0.0193, -0.0123] | -0.0379 [-0.0462, -0.0298] | +0.0176 [0.0121, 0.0233] |
| Policy and lifecycle | +0.0105 [0.0045, 0.0165] | +0.0170 [0.0030, 0.0305] | -0.1236 [-0.1348, -0.1119] |

The released-distractor policy axis drives the observed incremental benefit. The
two-intent lifecycle rule is too coarse and harms retrieval when applied alone. The
combined arm remains an oracle/released-field diagnostic, not a deployable metadata
inference method.

## Metadata Break-Even

For strict namespace support, record-level false-deny and missing-label corruption
still weakly dominate global dense at the observed 10% point on both feasible rate
and penalized admissibility upper risk. The first non-dominating grid point is 20%.
At 20% missingness, feasible rate is 0.5094 versus 0.5389 for global dense, while
risk remains lower (0.5622 versus 0.6276); the utility condition fails first.

Governance errors are asymmetric. Relative to namespace dense:

- policy false-allow and fail-open missing remain weakly dominant through 30%,
  with 40% the first non-dominating point;
- policy false-deny crosses the weak-dominance boundary at 2%;
- lifecycle false-stale also crosses the weak-dominance boundary at 2%;
- lifecycle false-current and fail-open lifecycle missing do not lose dominance by
  50%, because they undo over-filtering by the coarse released two-intent rule; and
- intent flips do not erase the policy-driven combined-arm advantage by 50%.

The last result is not evidence that lifecycle corruption is beneficial in general.
It diagnoses misspecification in this released-field lifecycle approximation.

## Interpretation Boundaries

Source-required evidence defines recall anchors even when its released lifecycle
field is unresolved. This preserves the source gold and prevents a coarse metadata
field from cancelling a required reference.

Non-anchor relevance is mostly unresolved in these sources. Therefore the observed
`known_relevant_inadmissible_rate=0` is a construction boundary, not evidence that
relevant-but-inadmissible memories have zero prevalence. The admissibility-only
bounds remain informative; the full relevance-by-admissibility prevalence does not.

The full-population natural corruption run covers strict namespace false-deny and
missing labels plus policy, lifecycle, and intent errors. Namespace false-allow and
label-swap expand support across namespaces and require a new global reranking run;
they remain covered only by the executable synthetic protocol and are not claimed as
full natural-corpus results here.

Run `python -m scripts.publish_supplemental_results verify` to validate every data
file and provenance receipt in this directory.
