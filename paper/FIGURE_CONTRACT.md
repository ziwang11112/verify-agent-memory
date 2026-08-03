# Figure Contract

This file fixes the message, evidence population, and claim boundary for every main
paper figure. It is a review aid; `claims/claims.yaml` remains authoritative.

## Figure 1: The Dangerous Memory Is Relevant but Ineligible

**Conclusion:** Topical overlap does not determine whether a retrieved record may
support the current query. Scope, policy, and lifecycle must be checked against the
query; an old state can be excluded for a current-state query yet retained for a
history query.

| Panel | Role | Intended read |
| --- | --- | --- |
| a | Same-memory MemOps counterfactual | A superseded date is retained for a history query and rejected for a current-state query |
| b | Relevance-by-admissibility matrix | Relevant-inadmissible evidence is the dangerous quadrant; unresolved status is never silently passed |
| c | Enforcement path and paper map | Verification occurs after retrieval but before prompt exposure; Figures 2--4 diagnose support, verifier, and reader failures |

**Evidence:** C1 and the abridged superseded-Lisbon record from the audited MemOps
packet listed in `evidence/examples/retrieval_admissibility_cases.json`. The history
query is an example-derived counterfactual. The other three audited cases appear in
Appendix Figure A1. These are qualitative examples, not frequency or effect estimates.

**Boundary:** Text is abridged for figure legibility; packet IDs, memory IDs, source
labels, and verdicts are preserved in the example manifest. The figure defines the
verification target and required observations. It does not infer policy, scope,
intent, or lifecycle state, and it does not estimate population prevalence.

## Figure 2: Constraint Value Depends on Metadata Reliability

**Conclusion:** Trusted namespace support helps semantic ranking at practical
budgets, but the gain is not a generic consequence of metadata. Policy supplies the
incremental governance benefit in the released-field approximation, while false
denies and source-label swaps erase utility earlier than fail-open errors.

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 87 groups and 3,767 public-source queries | Global-to-namespace movement in the absolute recall--penalized-admissibility-risk plane at top-$k\in\{10,20,50,100\}$; selected annotations give feasibility deltas | Trusted support moves every budget toward higher recall and lower risk |
| b | Same frozen settings and population | Policy-only, lifecycle-only, and corrected governance-v2 points in the risk-reduction--recall-gain plane | Released policy metadata drives the incremental gain; the coarse lifecycle-only approximation over-filters |
| c | Same population; 10 nested seeds per corruption channel | Joint utility--risk dominance over observed corruption rates, plus wrong-scope exposure and candidate work | Error direction matters: false denial and source-label swap damage utility sooner; fail-open support still leaks and expands work before aggregate dominance disappears |

**Boundaries:** Top-$k$ and attribution intervals use 10,000 source-clustered
bootstrap replicates. Corruption points are ten-seed means; observed break-even
brackets are grid diagnostics, not interpolated thresholds. The historical v1
non-usable selection score is not used as an admissibility-only outcome. This is not
an official RHELM or MemOps submission.

## Figure 3: Semantic Verifiers Recognize Rules but Cannot Apply Them Selectively

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 72 public-development cases; fixed top-20 candidates | Released-oracle and text-inferred feasibility versus sign-reversed admissibility-risk differences, with recall deltas | Released fields occupy the useful quadrant; text-only filters fail to realize the headroom and can worsen both outcomes |
| b | 16 controlled scenarios and 32 focal pairs | Focal consistency versus stable-control overflip, with stable-admissible false denial | Strong focal rule-following coexists with over-filtering of memories whose status should remain unchanged |

**Boundaries:** Panels a and b are separate public-development populations and are
never pooled. OpenAI and Gemini are comparison-eligible in the natural diagnostic;
OpenAI, Gemini, and DeepSeek are complete in the controlled diagnostic. Released
metadata is an oracle reference, not a deployable method. The result does not cover
latent production state or untested models. Candidate-level ROC-AUC and operating-
point diagnostics are reported in the appendix rather than mixed into the main plot.

## Figure 4: Exposure Converts Residual Verification Errors into Disclosure

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 16 controlled scenarios; 192 paired units per reader | Exposed-minus-withheld disclosure effects in all four relevance-by-admissibility cells | The empirical matrix closes the loop with Figure 1: the largest effects occur for relevant admissible evidence |
| b | Same paired population | Relevant-admissible minus relevant-inadmissible effect, with the residual inadmissible effect shown separately | All readers have a positive selectivity gap, while DeepSeek retains a positive inadmissible exposure effect |

**Boundaries:** The three readers are reported separately and never pooled. Whiskers
are 95% scenario-bootstrap intervals. The estimand is assigned exposure under
construction-defined admissible and inadmissible query conditions, not an isolated
causal effect of admissibility, internal model use, natural prevalence, production
safety, or an official benchmark result.

## Appendix Figure A1: Additional Audited Admissibility Cases

**Conclusion:** Wrong scope, supersession, explicit forgetting, and historical intent
produce distinct eligibility decisions that the main same-memory example does not
exhaust. These examples establish taxonomy coverage, not prevalence.

## Appendix Figure A2: Frozen Nine-Arm Retrieval Diagnostics

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 87 groups and 3,767 public-source queries | Recall versus candidates scored for all nine frozen v1 arms | Namespace support, not the evaluated routing refinements, separates the main regimes |
| b | Feasible matched-recall prefixes | Historical non-usable known rate, coverage, and bounds | The preregistered v1 score combines relevance and admissibility and must retain that name |
| c | Same population | Threshold and cluster differences versus namespace dense | Neither frozen router establishes incremental retrieval utility |

**Boundaries:** The released-intent lifecycle arm in this figure is frozen historical
v1. It is not the corrected governance-v2 attribution arm and cannot support a
lifecycle-only claim.

## Appendix Figure A3: Natural Route Traces Are Observational

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 721 GateMem exposure-discordant checkpoints | Exposed-minus-unexposed answer-disclosure risk difference | Exposure is strongly associated with answer disclosure for each reader |
| b | 4,470 held-out GateMem rows | Exposure-augmented minus baseline Brier score and log loss | Exposure improves held-out prediction under two proper scores |
| c | 1,490 leakage and 728 distinct bounded-utility checkpoints | G1-minus-G0 leakage, utility, and over-refusal | Stricter filtering exposes a safety--utility frontier |

**Boundaries:** GateMem readers are same-provider and never pooled. Exposure/disclosure
and G1/G0 route comparisons are associative, not causal. Leakage and utility
populations differ. This figure complements but does not share a population or
estimand with Figure 2.

## Rendering

- Backend: Python/Matplotlib.
- Main-text width: 7.1 inches.
- Archival outputs: PDF and SVG.
- Visual-inspection output: PNG.
- Error bars: 95% intervals from frozen evidence.
- Zero reference: shown whenever a panel reports a delta.
- Status colors: green for admissible/benefit, red for inadmissible/risk, gray for
  unresolved or global support, and teal for trusted namespace support.
- Model colors: blue for GPT-5.6, orange for Gemini, and purple for DeepSeek in every
  model-comparison figure.
