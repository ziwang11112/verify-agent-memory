# Figure Contract

This file fixes the message, evidence population, and claim boundary for every main
paper figure. It is a review aid; `claims/claims.yaml` remains authoritative.

## Figure 1: Topical Match Is Not Eligible Evidence

**Conclusion:** Semantic retrieval can surface memories from the wrong principal,
superseded state, or an explicit forget target; authenticated eligibility evidence
must decide which records may cross the prompt boundary. The same old record can be
excluded for a current-state query and retained for a history query.

| Panel | Role | Intended read |
| --- | --- | --- |
| a | Compact query--candidate audit matrix over four RHELM/MemOps rows | Wrong-principal, superseded-current, and explicitly forgotten candidates can be topically matched yet fail different pre-prompt checks; a history row shows that old state is not intrinsically inadmissible |
| b | Verification path and three evidence blocks | Support and end-to-end cards distinguish the full natural retrieval audit from its frozen 1,523-case route-to-answer subset; the exposure card is a separate controlled population |

**Evidence:** Panel a uses C1 and the abridged wrong-namespace market,
superseded-Lisbon, and user-forgotten-phone cases listed in
`evidence/examples/retrieval_admissibility_cases.json`; the Lisbon history note is an
audited public-source example. Panel b indexes C9 top-20 support, C13 natural closure,
and the C8/C12 controlled exposure executions. The support and closure cards share the
frozen natural source but have different sample sizes; controlled exposure remains
separate. All four audited cases appear in Appendix Figure 4.

**Boundary:** Text is abridged for figure legibility; packet IDs, memory IDs, source
labels, and verdicts are preserved in the example manifest. The examples do not infer
policy, scope, intent, or lifecycle state, estimate prevalence, or pool the roadmap
statistics or readers.

## Figure 2: Constraint Value Depends on Metadata Reliability

**Conclusion:** Trusted namespace support helps semantic ranking at practical
budgets, but the gain is not a generic consequence of metadata. Policy supplies the
incremental governance benefit in the released-field approximation, while false
denies and source-label swaps erase utility earlier than fail-open errors.

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 87 groups and 3,767 public-source queries | Global-to-namespace movement in the absolute recall--penalized-admissibility-risk plane at top-$k\in\{10,20,50,100\}$; selected annotations give feasibility deltas | Trusted support moves every budget toward higher recall and lower risk |
| b | Same frozen settings and population | Policy-only, lifecycle-only, and corrected governance-v2 points in the risk-reduction--recall-gain plane | Released policy metadata drives the incremental gain; the coarse lifecycle-only approximation over-filters |
| c | Same population; 10 nested seeds per corruption channel | Channel-specific weak-dominance brackets over observed corruption rates; namespace channels use global dense and governance channels use clean namespace dense | Error direction matters: false denial and source-label swap damage utility sooner; fail-open support still leaks and expands work before aggregate dominance disappears |

**Boundaries:** Top-$k$ and attribution intervals use 10,000 namespace-group
bootstrap replicates under equal-source macro-averaging. Corruption points are
ten-seed means; observed break-even brackets are grid diagnostics, not interpolated
thresholds. The historical v1
non-usable selection score is not used as an admissibility-only outcome. This is not
an official RHELM or MemOps submission.

## Figure 3: Controlled Exposure Can Enable Literal-Marker Disclosure

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 16 controlled scenarios; 192 paired units per reader in each complete execution | Exposed-minus-withheld disclosure effects in all four relevance-by-admissibility cells | The empirical matrix closes the loop with Figure 1: the largest effects occur for relevant admissible evidence |
| b | Same construction; original three-reader execution plus a separate Claude replication | Relevant-admissible minus relevant-inadmissible effect, with the relevant-inadmissible effect shown separately | All four readers have a positive selectivity-gap interval; only DeepSeek has a strictly positive relevant-inadmissible interval |

**Boundaries:** GPT-5.6 Sol, Gemini 3.6 Flash, and DeepSeek V4 Pro come from the
original execution. Claude Opus 5 is an independently executed C12 replication and
is not retroactively inserted into or pooled with C8. Every reader is reported
separately. Whiskers are 95% scenario-bootstrap intervals. The estimand is assigned
exposure under construction-defined admissible and inadmissible query conditions,
not an isolated causal effect of admissibility, internal model use, natural
prevalence, production safety, or an official benchmark result.

## Appendix Figure 4: Additional Audited Admissibility Cases

**Conclusion:** Wrong scope, supersession, explicit forgetting, and historical intent
produce distinct eligibility decisions that the main examples do not exhaust. These
examples establish taxonomy coverage, not prevalence.

## Appendix Figure 5: Semantic Verifiers Recognize Rules but Cannot Apply Them Selectively

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

## Appendix Figure 6: Frozen Nine-Arm Retrieval Diagnostics

| Panel | Population | Quantity | Intended read |
| --- | --- | --- | --- |
| a | 87 groups and 3,767 public-source queries | Recall versus candidates scored for all nine frozen v1 arms | Namespace support, not the evaluated routing refinements, separates the main regimes |
| b | Feasible matched-recall prefixes | Historical non-usable known rate, coverage, and bounds | The preregistered v1 score combines relevance and admissibility and must retain that name |
| c | Same population | Threshold and cluster differences versus namespace dense | Neither frozen router establishes incremental retrieval utility |

**Boundaries:** The released-intent lifecycle arm in this figure is frozen historical
v1. It is not the corrected governance-v2 attribution arm and cannot support a
lifecycle-only claim.

## Appendix Figure 7: Natural Route Traces Are Observational

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
- Model colors: blue for GPT-5.6, orange for Gemini, purple for DeepSeek, and dark
  teal for Claude in every model-comparison figure that includes those readers.
