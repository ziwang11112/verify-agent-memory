# Figure Contract

This file fixes the message, evidence population, and claim boundary for every main
paper figure. It is a review aid; `claims/claims.yaml` remains authoritative.

## Figure 1: Verification Pipeline

**Conclusion:** Retrieval verification must distinguish stored, retrieved, exposed,
and used events, and must evaluate relevance separately from query-conditioned
admissibility.

**Evidence:** Conceptual definition C1; no empirical population.

**Boundary:** The diagram does not infer policy, scope, intent, or lifecycle state.

## Figure 2: Exposure Is Informative, but Filtering Is Costly

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 721 GateMem exposure-discordant checkpoints | Exposed-minus-unexposed answer-use risk difference | Exposure is strongly associated with answer use for each reader |
| b | 4,470 held-out GateMem rows | Exposure-augmented minus baseline Brier score | Exposure improves held-out prediction; lower is better |
| c | 4,470 held-out GateMem rows | Exposure-augmented minus baseline log loss | The predictive result is not peculiar to one proper score |
| d | 1,490 leakage and 728 distinct bounded-utility checkpoints | G1-minus-G0 leakage, utility, and over-refusal | Stricter filtering exposes a safety--utility frontier |
| e | 207 selected records | Raw exact agreement and Krippendorff alpha | Most axes are reliable enough for audit; prohibited is prevalence-limited |

**Boundaries:** Readers are same-provider and never pooled. Exposure/use and G1/G0
comparisons are associative, not causal. Leakage and utility populations differ.
Human labels are raw, pre-adjudication, and not population-wide gold.

## Figure 3: Namespace Support Carries the Retrieval Gain

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 87 groups, 3,767 public-source queries | Absolute global- and namespace-dense evidence recall | Trusted namespace support improves the primary retrieval outcome |
| b | Same public-source population | Namespace-dense minus global-dense deltas | Namespace support improves recall and feasibility while lowering the conservative contamination upper bound |
| c | Same public-source population | Threshold/cluster minus namespace-dense deltas | Neither frozen router establishes incremental utility |
| d | Same population; 2,764 eligible lifecycle-exposure rows | Released-field arm minus namespace dense | Observed intent and lifecycle fields reveal upper-bound headroom |

**Boundaries:** Results are source-macro public-source diagnostics, not official
RHELM or MemOps submissions. Namespaces are trusted and released, not inferred.
There is no downstream reader result for this population. Lifecycle is an observed-
field upper bound, not a deployable classifier.

## Rendering

- Backend: Python/Matplotlib.
- Main-text width: 7.1 inches.
- Archival outputs: PDF and SVG.
- Visual-inspection output: PNG.
- Error bars: 95% intervals from frozen evidence.
- Zero reference: shown whenever a panel reports a delta.
