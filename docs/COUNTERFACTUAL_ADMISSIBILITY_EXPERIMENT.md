# Counterfactual Admissibility Pairs

This controlled public-development diagnostic tests a narrower question than the
natural-source audit: can a verifier change an admissibility decision when the query's
governing condition changes, while leaving a fixed candidate pool untouched?

It addresses a specific ambiguity in ordinary retrieval results. A model can appear to
recognize stale or wrong-scope evidence because the two memories discuss different
topics. Counterfactual pairs hold candidate text and order fixed and alter only one
query-side condition. The experiment therefore tests query-conditioned verification,
not retrieval quality and not latent metadata recovery.

## Frozen Design

The dataset contains 16 independent scenarios, four for each axis:

| Axis | Counterfactual change |
| --- | --- |
| `principal_scope` | authorized requester or namespace versus an unauthorized one |
| `lifecycle_intent` | historical-use intent versus current-state intent |
| `policy_purpose` | an allowed use purpose versus a prohibited purpose |
| `as_of_time` | a date inside versus outside an explicit validity interval |

Every scenario has two paraphrase pairs. This yields 32 pairs and 64 condition cases.
The two cases in a pair receive the same three candidates in the same order:

- one focal candidate that is admissible under the `allow` query and inadmissible under
  the `block` query;
- one stable-admissible control that should remain admissible; and
- one stable-inadmissible control that should remain inadmissible.

All required scope, policy, lifecycle, and time evidence is explicit in the public
controlled text. This makes the diagnostic a test of rule application under minimal
query changes. It must not be presented as evidence that models can infer missing
governance metadata in natural conversations. Four scenarios are abridged from the
audited public examples; the remainder are controlled synthetic analogues fixed before
provider execution.

Each query pair must have token-set Jaccard similarity of at least 0.5. Candidate roles,
gold labels, scenario identity, axis, and source basis are scorer-only fields. The model
sees only the query and the fixed candidate keys and text.

## Metrics

The primary metric is strict focal-pair consistency: the focal candidate must be called
`admissible` on the allow side and `inadmissible` on the block side. A one-sided correct
answer receives no pair credit. The symmetric directional margin is

```text
score(q, m) = p(admissible) - p(inadmissible)
margin(pair) = [score(allow, focal) - score(block, focal)] / 2.
```

Stable-control accuracy checks ordinary classification, while stable-control overflip
measures whether a model changes answers merely because the query wording changed.
Candidate accuracy, multiclass Brier score, unknown rates, and lexical similarity are
secondary diagnostics.

Results are reported globally and by axis. Confidence intervals use 10,000 bootstrap
replicates, resampling whole scenarios within each axis and averaging axes equally. The
two paraphrases from a scenario are never treated as independent bootstrap units.

The released-label oracle is a ceiling. `no_verifier_keep_all` represents semantic
retrieval without an admissibility verifier: it emits the same admissible decision for
both sides and therefore cannot solve the focal flip.

## Interpretation Boundary

The preregistered supportive pattern is at least 0.80 global strict pair consistency,
at least 0.75 on every axis, stable-control overflip no greater than 0.05, and positive
mean directional margin. These thresholds organize this diagnostic; they are not an E6
or M2 gate.

Even a positive result establishes only that a model can apply explicit visible
conditions in controlled pairs. A negative result would show that the proposed verifier
is not reliably query-conditioned even in that favorable setting. Neither outcome is an
official benchmark result, a reader-safety result, or evidence about hidden policy and
lifecycle inference.

## Provider Panel and Cost Boundary

The frozen panel reuses the exact four bindings from the preceding inferred-
admissibility diagnostic: GPT-5.6 Sol, DeepSeek-V4-Pro, Gemini 3.6 Flash, and Claude
Sonnet 5. The output is much smaller: three direct admissibility vectors and no answer,
reader, or judge. Provider comparison requires all 64 cases; incomplete prefixes are
retained for execution audit but not scored.

The protocol has a total hard cap of USD 7.50, including provider fixtures. Execution
is append-only, checkpointed after every case, and forbids semantic repair, selective
reruns, held-out access, or benchmark claims.
