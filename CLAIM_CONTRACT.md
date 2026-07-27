# Claim Contract

This document renders the machine-readable contract in `claims/claims.yaml`. The
YAML file is authoritative for automated checks; this document is the readable
paper-writing boundary.

## Formal Distinction

### C1: Retrieval admissibility

**Status:** Definition

**Claim:** A memory can be relevant and still be inadmissible for the current query.

```text
admissible(m, q, p, t)
  = scopeAllowed(m, q, p)
    AND policyAllowed(m, q, p)
    AND lifecycleCompatible(m, intent(q), t)

usable(m, q, p, t)
  = relevant(m, q)
    AND admissible(m, q, p, t)
```

**Allowed:** Admissibility is conditioned on the query, policy, lifecycle, and
time.

**Forbidden:** Do not equate every inadmissible memory with security harm. Do not
claim that stale or superseded memories are never useful.

**Boundary:** The definition is not an inference procedure for latent intent or
policy.

## Association

### C2: Exposure and answer leakage

**Status:** Frozen association

**Population:** 721 exposure-discordant privacy or safety checkpoints from a frozen
GateMem route bundle.

| Reader | Risk difference | 95% CI | Brier delta | Log-loss delta |
| --- | ---: | ---: | ---: | ---: |
| `gpt-4o-mini-2024-07-18` | 0.4286 | [0.3919, 0.4642] | -0.022532 | -0.052176 |
| `gpt-4o-2024-08-06` | 0.5298 | [0.4915, 0.5665] | -0.025471 | -0.063268 |

Table values are rounded for display; the YAML contract and normalized evidence
retain source precision.

The Brier and log-loss intervals are recorded in the YAML contract. Both models are
OpenAI readers. The estimates are reported separately and are never pooled.

**Allowed:** Across fixed checkpoints and separately executed readers, exposing the
protected target was strongly associated with whether it appeared in the answer.
Adding exposure improved held-out leakage prediction for each reader.

**Forbidden:** Do not call this causal mediation, a randomized treatment effect, a
cross-provider replication, a pooled estimate, or an official GateMem result.

**Boundary:** This is same-provider cross-reader robustness and a non-causal
association.

## Leakage-Utility Trade-Off

### C3: Reducing exposure has a utility cost

**Status:** Frozen trade-off

The paired G1-minus-G0 comparison produced:

| Metric | Difference | 95% CI | Eligible checkpoints |
| --- | ---: | ---: | ---: |
| Answer leakage | -0.220134 | [-0.249196, -0.191904] | 1,490 |
| Utility accuracy | -0.168956 | [-0.207922, -0.130790] | 728 |
| Over-refusal | 0.287088 | [0.233783, 0.340488] | 728 |

**Allowed:** Reducing exposure lowered leakage but incurred a bounded utility and
over-refusal cost. The result motivates a safety-utility verification problem.

**Forbidden:** Do not say the stricter policy is uniformly better, that filtering is
free, or that this comparison identifies a causal treatment effect.

**Boundary:** Leakage and bounded utility use different eligible populations. The
estimate comes from one OpenAI reader and is not cross-provider evidence.

## Mechanism Smoke

### C4: Candidate-pool scorer check

**Status:** Mechanism smoke only

The audit contained four development packets and 16 evaluation packets, with 207
records across the full audit and 7-12 candidates per packet.

| Method | Recall | Wrong-namespace leakage | Measured contamination |
| --- | ---: | ---: | ---: |
| Global dense | 1.0000 | 0.0250 | 0.1189 |
| Namespace dense | 1.0000 | 0.0000 | 0.0824 |

Table values are rounded for display; the YAML contract and normalized evidence
retain source precision.

**Allowed:** The small candidate-pool smoke verifies that the scorer responds to a
known scope constraint and detects recall loss from aggressive routing.

**Forbidden:** Do not present these 16 packets as a natural-corpus estimate, an
official benchmark result, or part of the same population as the 3,767-query
evaluation.

**Boundary:** Candidate pools were curated and development selection used only four
packets. No reader or answer outcome was evaluated.

## Full Natural-Corpus Evaluation

### C5: Trusted namespace support

**Status:** Frozen public-source evaluation

The evaluation applied settings selected once on development data to 87 groups,
182,908 memories, and 3,767 queries.

| Quantity | Estimate | 95% CI |
| --- | ---: | ---: |
| Global-dense recall | 0.716835 | n/a |
| Namespace-dense recall | 0.865663 | n/a |
| Recall difference | 0.148828 | [0.129460, 0.168372] |
| Feasible-rate difference | 0.222996 | [0.192156, 0.255019] |
| Conservative-contamination difference | -0.035958 | [-0.041611, -0.030301] |
| Namespace wrong-scope leakage | 0.000000 | n/a |

**Allowed:** Trusted namespace support improved recall and feasible rate while
reducing the conservative contamination upper bound on this public-source
evaluation. Namespace arms had zero measured wrong-scope leakage under trusted
released namespaces.

**Forbidden:** Do not call this an official RHELM or MemOps submission, claim a
downstream answer-quality effect, generalize to noisy or inferred namespaces, or
describe the conservative bound as fully resolved contamination.

**Boundary:** Non-required same-namespace memories remain unresolved. There is no
reader outcome for this population. Intervals are source-macro bootstrap intervals.

## Diagnostic Routing

### C6: No established value beyond trusted namespace

**Status:** Diagnostic negative result

| Contrast against namespace dense | Recall difference | 95% CI | Conservative contamination difference | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Threshold router | -0.023001 | [-0.028848, -0.017318] | 0.002100 | [0.001476, 0.002854] |
| Cluster router | -0.000609 | [-0.001875, 0.000000] | 0.000166 | [-0.000001, 0.000513] |

**Allowed:** Additional threshold and cluster routing did not establish practical
incremental utility beyond trusted namespace support. The threshold arm lost recall;
the cluster arm was practically indistinguishable on primary quality metrics.

**Forbidden:** Do not present a new router as the contribution, claim clustering is
universally useless, or claim exact identity on every full-evaluation query.

**Boundary:** This result covers two frozen diagnostic variants, not the class of all
possible routing methods.

## Upper-Bound Result

### C7: Released intent and lifecycle headroom

**Status:** Released-field upper bound

| Upper-bound minus namespace dense | Difference | 95% CI |
| --- | ---: | ---: |
| Recall | 0.002902 | [-0.002336, 0.007897] |
| Conservative contamination | -0.008667 | [-0.010697, -0.006750] |
| Prohibited-stale exposure | -0.002475 | [-0.003401, -0.001653] |
| Prohibited-superseded exposure | -0.010397 | [-0.011768, -0.009086] |

**Allowed:** Released query intent and lifecycle fields reveal upper-bound headroom
without a detected recall loss.

**Forbidden:** Do not describe this as a deployable lifecycle filter, claim the
system inferred intent or state, or recommend deleting every stale memory.

**Boundary:** Query intent and lifecycle state are released inputs. Historical
queries may require records that are stale for current-state queries.

## Human Agreement Boundary

### C8: Raw two-human overlap

**Status:** Raw pre-adjudication agreement

Two independent humans labeled 207 selected records. Model-assisted labels are
excluded.

| Axis | Exact agreement | Krippendorff alpha |
| --- | ---: | ---: |
| Relevance | 0.8937 | 0.8135 |
| Scope | 0.9952 | 0.9808 |
| State | 0.8599 | 0.6703 |
| Prohibited | 0.8841 | 0.0829 |
| Usable evidence | 0.9710 | 0.9405 |

**Allowed:** Agreement was strong for relevance, scope, and usable evidence, and
moderate for state. The prohibited axis had high exact agreement but low alpha under
strong prevalence imbalance.

**Forbidden:** Do not summarize all five axes as highly reliable, call the selected
records population-wide human gold, or count model-assisted labels as humans.

**Boundary:** The overlap is raw and pre-adjudication. A distinct execution commit is
not recorded in the public summary and must be resolved or explicitly preserved as a
provenance gap before Phase 2.

## Writing Rule

Every manuscript sentence that asserts an empirical result must map to one claim ID.
When a sentence combines claims, it must preserve the population, estimand, and
limitations of each source claim. Mechanism-smoke values and full-evaluation values
must never be pooled, averaged, or narrated as one experiment.
