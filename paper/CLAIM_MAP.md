# Paper Claim Map

This map is a writing and review gate. `claims/claims.yaml` remains authoritative.

| Claim | Paper location | Artifact | Required boundary |
| --- | --- | --- | --- |
| C1 | Introduction and Section 3.1 | Figure 1 | Abridged audited examples plus a definition; not a latent-intent inference procedure or prevalence estimate |
| C2 | Sections 3.4, 4.2, and 5.2 | Appendix Figure 4a--b | Association only; same-provider readers; no pooled estimate |
| C3 | Sections 3.4, 4.2, and 5.2 | Appendix Figure 4c, Table 2 | Leakage reduction has bounded utility and over-refusal costs |
| C4 | Section 4.4 and Appendix D | Table 6 | Sixteen-packet mechanism smoke, not a natural-corpus estimate |
| C5 | Sections 3.2--3.3, 4.5--4.6, and 5.3 | Figure 3a--c, Tables 2--3 | Trusted released namespaces; not an official benchmark submission |
| C6 | Sections 4.5--4.6 and 5.4 | Figure 3d | Negative result for two frozen diagnostic routers only |
| C7 | Sections 3.1, 4.5--4.6, and 5.5 | Figure 3d, Table 2 | Released-field upper bound, not deployable blind inference |
| C8 | Sections 3.4, 4.3, and 5.1 | Figure 2, Table 2 | Controlled constructed population; literal disclosure rather than internal use; readers never pooled |

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
6. Natural-route exposure remains associational, while a separate paired intervention
   identifies a controlled prompt-level disclosure effect in a constructed population.
7. Reader selectivity does not replace pre-prompt admissibility verification.

## Claims Not Made

- A new state-of-the-art memory index or router.
- An official GateMem, RHELM, or MemOps benchmark result.
- Generalization from trusted namespaces to noisy or inferred namespaces.
- A causal effect of natural retrieval routes on answer leakage; C8 is limited to its
  paired prompt-level intervention.
- Cross-provider replication of the GateMem reader analysis.
- A deployable method for inferring intent, scope, or lifecycle state.
