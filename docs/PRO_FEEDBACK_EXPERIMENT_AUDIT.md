# Pro Feedback Experiment Audit

This document maps the prior external review to executable evidence in this
repository. It is an implementation-status audit, not manuscript prose and not a new
empirical claim.

## Status Map

| Review item | Status | Repository evidence | Remaining action |
| --- | --- | --- | --- |
| Separate relevance error from admissibility violation | Complete in metric schema v2 | `metrics.py`, `experiment.py`, `natural_admissibility_attribution.csv` | Promote only after the claim contract binds the supplemental result |
| Decompose penalized risk into infeasibility and conditional risk | Complete | `experiment.py`, supplemental natural tables | Preserve both components in every paper table |
| Apply policy independently of lifecycle intent | Complete | `released_intent_lifecycle_upper_bound` and policy-only attribution arm | Keep the arm labeled as a released-field upper bound |
| Make policy query-conditioned | Complete as a released approximation | Query-memory `PolicyDecision` and `PolicyPurpose` in `retrieval.py` | Do not claim a general policy engine |
| Replace binary current/history semantics with general temporal validity | Bounded, not solved | Two-intent limitation plus controlled as-of-time pairs | Keep natural-corpus lifecycle claims explicitly approximate |
| Metadata corruption and break-even | Complete | `metadata_robustness_protocol.json`, `natural_*_break_even.json` | Move the result into the claim contract before manuscript use |
| Oracle, inferred, unknown, and no-metadata comparison | Complete, negative for text inference | `results/inferred_admissibility/` | Bind as a diagnostic claim; do not present the oracle as deployable |
| Counterfactual admissibility pairs | Complete | `results/counterfactual_admissibility/` | Bind the primary and posthoc boundaries separately |
| Practical top-k frontier | Complete | `natural_top_k_pareto.csv` | Use fixed-budget comparisons instead of emphasizing top-k 100 alone |
| Same-population stage trace | Substantially complete on GateMem | Frozen routes, assembled prompts, and fresh reader answers for the same 2,218 checkpoints | State that route differences remain observational interventions |
| Randomized or paired exposure intervention | Zero-call design complete; empirical run pending | `counterfactual_exposure_protocol.json`, pure construction/scoring module, and zero-call tests | Freeze exact reader snapshots and cost caps before any paid execution |
| Direct RaMem comparison | Not comparable on the current governance populations | RaMem supports LoCoMo and LongMemEval-S, while this audit targets scope, policy, and lifecycle labels | Cite and contrast; do not port it as a nominal baseline without a common estimand |

## What Is Already Established

The completed work supports a qualified pattern rather than a new retrieval
algorithm:

1. Correct support constraints dominate the tested semantic routing refinements when
   released governance metadata is sufficiently reliable.
2. That advantage loses joint recall-risk dominance as metadata errors increase, and
   false-deny governance errors are especially costly.
3. Released fields show oracle headroom, but text-only inference does not realize it
   on the frozen public-development sample.
4. Strong verifiers can classify the focal counterfactual correctly while changing
   stable candidates that should not change, primarily through false denial.

## Remaining Empirical Gap

The unanswered empirical question is whether controlled exposure of a memory changes
the final answer differently in the four relevance-by-admissibility cells. Existing
GateMem route comparisons observe full retrieval-to-answer traces, but route policies
alter more than one piece of context and exposure is not assigned independently. The
new zero-call protocol therefore manipulates one candidate at a time while holding
the query and all other candidates fixed. Its dataset contract, request construction,
literal scorer, and bootstrap are implemented and tested; no reader result exists
until a separately approved execution is complete.

No additional clustering, Bayesian indexing, broad public benchmark, or nominal
RaMem port is prioritized. Those additions would increase surface area without
addressing the remaining identification gap.
