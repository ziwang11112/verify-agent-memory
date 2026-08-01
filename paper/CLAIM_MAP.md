# Paper Claim Map

This map is a writing and review gate. `claims/claims.yaml` remains authoritative.

| Claim | Paper location | Artifact | Required boundary |
| --- | --- | --- | --- |
| C1 | Introduction and Section 3.1 | Figure 1 | Abridged audited examples plus a definition; not a latent-intent inference procedure or prevalence estimate |
| C2 | Sections 4.1, 4.3, and 5.1 | Figure 2a--b, Table 1 | Association only; same-provider readers; no pooled estimate |
| C3 | Sections 4.1, 4.3, and 5.2 | Figure 2c, Table 1 | Leakage reduction has bounded utility and over-refusal costs |
| C4 | Appendix B | Table 2 | Sixteen-packet mechanism smoke, not a natural-corpus estimate |
| C5 | Sections 4.1--4.3 and 5.3 | Figure 3a--c, Table 1 | Trusted released namespaces; not an official benchmark submission |
| C6 | Sections 4.2 and 5.4 | Figure 3d, Table 1 | Negative result for two frozen diagnostic routers only |
| C7 | Sections 4.2 and 5.5 | Figure 3d, Table 1 | Released-field upper bound, not deployable blind inference |
| C8 | Sections 4.1, 4.3, and 5.6 | Figure 2d, Table 3 | Selected records; raw pre-adjudication agreement; prohibited-axis caveat |

## Paper-Level Claims

1. Retrieval verification should distinguish relevance from query-conditioned
   admissibility.
2. Stored, retrieved, exposed, and disclosed are different observable events and should
   not be collapsed into one success label.
3. Matched-recall contamination with explicit incomplete-label bounds provides a
   conservative retrieval diagnostic.
4. On the frozen public-source evaluation, trusted namespace support matters more
   than the evaluated threshold or cluster routing refinements.
5. Released lifecycle and intent fields show diagnostic headroom but do not establish
   a deployable lifecycle classifier.
6. Exposure-to-disclosure measurements motivate verification, but the observed association
   is not a causal effect.

## Claims Not Made

- A new state-of-the-art memory index or router.
- An official GateMem, RHELM, or MemOps benchmark result.
- Generalization from trusted namespaces to noisy or inferred namespaces.
- A causal effect of retrieval or exposure on answer leakage.
- Cross-provider replication of the GateMem reader analysis.
- Fully reliable prohibited-evidence annotation.
- A deployable method for inferring intent, scope, or lifecycle state.
