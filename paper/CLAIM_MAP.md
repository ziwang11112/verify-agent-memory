# Paper Claim Map

This map is a writing and review gate. `claims/claims.yaml` remains authoritative.

| Claim | Paper location | Artifact | Required boundary |
| --- | --- | --- | --- |
| C1 | Introduction and Section 3.1 | Figure 1 | Abridged audited examples plus a definition; not a latent-intent inference procedure or prevalence estimate |
| C2 | Section 3.4 and Appendix | Appendix Figure A2a--b | Association only; same-provider readers; no pooled estimate |
| C3 | Section 3.4 and Appendix | Appendix Figure A2c | Leakage reduction has bounded utility and over-refusal costs |
| C4 | Appendix | Mechanism-smoke table | Sixteen-packet implementation check, not a natural-corpus estimate |
| C5 | Sections 3.3 and Appendix | Appendix Figure A1 | Frozen v1 penalized non-usable risk; trusted namespaces; not pure admissibility |
| C6 | Section 5 and Appendix | Appendix Figure A1 | Negative result for two frozen diagnostic routers only |
| C7 | Section 3.3 and Appendix | Appendix Figure A1 | Frozen historical v1 semantics, not corrected v2 or lifecycle-only headroom |
| C8 | Sections 3.4, 4, and 5.4 | Figure 4 | Assigned exposure under constructed query conditions; no isolated causal moderator; readers never pooled |
| C9 | Sections 4 and 5.1 | Figure 2a | Post-hoc v2 scoring on frozen rankings; no retuning; source-clustered intervals |
| C10 | Sections 4 and 5.2 | Figure 2b--c | Tested corruption mechanisms only; observed grid brackets are not population thresholds |
| C11 | Sections 4 and 5.3 | Figure 3 | Natural and controlled public-development diagnostics are separate and never pooled |

## Paper-Level Claims

1. Retrieval verification should distinguish relevance from query-conditioned
   admissibility.
2. Stored, retrieved, exposed, and disclosed are different observable events and should
   not be collapsed into one success label.
3. Matched-recall non-usable and admissibility risk families with incomplete-label
   bounds prevent relevance error and governance violation from being conflated.
4. Trusted namespace support improves frozen semantic rankings at every tested
   practical budget, with both feasibility and feasible-prefix risk contributions.
5. Metadata error is directionally asymmetric: false denial and source-label swap
   erase joint utility--risk dominance earlier than tested fail-open errors.
6. Released policy metadata drives the corrected v2 gain; the evaluated coarse
   lifecycle-only approximation hurts retrieval.
7. Released governance fields expose headroom that text-only verifiers do not recover;
   controlled verifiers also overflip stable evidence.
8. Natural-route exposure remains associational, while a separate paired intervention
   identifies a controlled prompt-level disclosure effect in a constructed population.
9. Reader selectivity does not replace pre-prompt admissibility verification.

## Claims Not Made

- A new state-of-the-art memory index or router.
- An official GateMem, RHELM, or MemOps benchmark result.
- Generalization beyond the tested corruption mechanisms, observed grids, public-
  development inference cases, or frozen model snapshots.
- A causal effect of natural retrieval routes on answer leakage, or an isolated causal
  effect of admissibility itself; C8 is an assigned-exposure intervention under
  construction-defined query conditions.
- Cross-provider replication of the GateMem reader analysis.
- A deployable method for inferring intent, scope, or lifecycle state.
