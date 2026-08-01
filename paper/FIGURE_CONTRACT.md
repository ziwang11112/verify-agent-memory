# Figure Contract

This file fixes the message, evidence population, and claim boundary for every main
paper figure. It is a review aid; `claims/claims.yaml` remains authoritative.

## Figure 1: Concrete Retrieval Failures and the Verification Layer

**Conclusion:** Topical overlap does not determine whether a retrieved record may
support the current query. Scope, policy, and lifecycle must be checked against the
query; an old state can be excluded for a current-state query yet retained for a
history query.

| Panel | Role | Intended read |
| --- | --- | --- |
| a | RHELM wrong-namespace example | Market-related content from another principal remains inadmissible despite topical overlap |
| b | MemOps supersession example | A confirmed current value must displace an invalidated earlier value for a current-state query |
| c | MemOps forget example | The forget instruction remains usable while the prohibited detail is excluded |
| d | MemOps historical-query example | Superseded states remain usable when the query explicitly asks for a timeline |
| bottom strip | Record-level verification interface | Ranked IDs receive scope, policy, and query-conditioned lifecycle checks, a reasoned verdict, and an observable trace |

**Evidence:** C1 and four abridged records from the audited public-source packets
listed in `evidence/examples/retrieval_admissibility_cases.json`. The examples are
qualitative evidence, not a frequency or effect estimate.

**Boundary:** Text is abridged for figure legibility; packet IDs, memory IDs, source
labels, and verdicts are preserved in the example manifest. The figure defines the
verification target and required observations. It does not infer policy, scope,
intent, or lifecycle state, and it does not estimate population prevalence.

## Figure 2: Exposure Is Informative, but Filtering Is Costly

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 721 GateMem exposure-discordant checkpoints | Exposed-minus-unexposed answer-disclosure risk difference | Exposure is strongly associated with answer disclosure for each reader |
| b | 4,470 held-out GateMem rows | Exposure-augmented minus baseline Brier score and log loss | Exposure improves held-out prediction under two proper scores |
| c | 1,490 leakage and 728 distinct bounded-utility checkpoints | G1-minus-G0 leakage, utility, and over-refusal | Stricter filtering exposes a safety--utility frontier |

**Boundaries:** Readers are same-provider and never pooled. Exposure/disclosure and G1/G0
comparisons are associative, not causal. Leakage and utility populations differ.

## Figure 3: Namespace Support Carries the Retrieval Gain

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 87 groups, 3,767 public-source queries | Evidence recall versus mean candidates scored for all nine arms | The support change, not routing sophistication, separates the global and namespace regimes; candidate count is diagnostic, not production latency |
| b | Same public-source population | Namespace-dense minus global-dense deltas | Namespace support improves recall and feasibility while lowering penalized conservative risk |
| c | Feasible matched-recall prefixes | Known contamination, label coverage, and conditional lower/upper bounds | Resolved contamination falls, but lower label coverage keeps the namespace upper bound high |
| d | Same population; 2,764 eligible lifecycle-exposure rows | Benefit-oriented threshold, cluster, and released-field differences versus namespace dense | Threshold hurts recall, clustering is near zero, and observed lifecycle fields reveal upper-bound headroom |

**Boundaries:** Results are source-macro public-source diagnostics, not official
RHELM or MemOps submissions. Namespaces are trusted and released, not inferred.
There is no downstream reader result for this population. Lifecycle is an
observed-field upper bound, not a deployable classifier. Panel d reverses the sign of
lower-is-better risk and exposure quantities so that positive always denotes a
favorable change.

## Rendering

- Backend: Python/Matplotlib.
- Main-text width: 7.1 inches.
- Archival outputs: PDF and SVG.
- Visual-inspection output: PNG.
- Error bars: 95% intervals from frozen evidence.
- Zero reference: shown whenever a panel reports a delta.
- Color vocabulary: neutral gray for global support, green for trusted namespace,
  orange/blue for diagnostic routers, and purple for released-field upper bounds.
