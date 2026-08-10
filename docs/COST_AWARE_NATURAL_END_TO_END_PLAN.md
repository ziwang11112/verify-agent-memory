# Cost-Aware Natural End-to-End Evaluation Plan

Status: proposed; no provider execution or performance scoring is authorized by this
document.

Date: 2026-08-08

Repository baseline: `8432abe7a34c1b44c4a943bcdd150858ad43f6b9`

Execution update (2026-08-09): the staged Gemini/DeepSeek panel and blinded judge
completed first. A subsequent GPT-only amendment then ran `gpt-5.6-luna` on the same
frozen 1,523-case judge sample and three primary routes. Claude Haiku remained the
cross-provider judge and was not added as a reader. The GPT reader plus judge cost
`$21.778532`; results are under
[`results/natural_end_to_end_gpt_luna_judged/`](../results/natural_end_to_end_gpt_luna_judged/).
The Stage 3 Luna-plus-Haiku-reader text below is retained as the historical planning
boundary and was not executed as written.

## 1. Decision summary

The same-population natural route-to-reader question remains scientifically useful,
but the frozen v2 execution is too broad and too expensive as a first test. Keep the
completed verifier bundle, preserve all partial reader checkpoints unscored, and do
not resume the four-reader full run.

Replace it with a staged evaluation:

1. zero-call source-gold and route audit;
2. a hash-bound two-reader amendment that preserves and completes the existing
   Gemini 3.6 Flash and DeepSeek V4 Pro checkpoints;
3. zero-cost deterministic scoring followed by a low-cost blinded judge only if the
   deterministic gate passes; and
4. full-population GPT-5.6 Luna and Claude Haiku replication only if the two-reader
   result passes utility, disclosure, and over-refusal gates.

The objective is not to find a favorable slice. It is to test, on one natural
population, whether pre-prompt support constraints reduce inadmissible exposure and
answer disclosure without materially reducing answer utility.

## 2. What the current evidence already establishes

| Evidence block | Current result | Defensible implication |
| --- | --- | --- |
| Natural support, 3,767 queries | Namespace dense changes recall by `+0.1488`, feasible rate by `+0.2230`, and penalized upper risk by `-0.0360` against global dense | Trusted namespace support improves the measured retrieval utility-risk frontier |
| Fixed-budget natural analysis | At top-20, recall changes from `0.4316` to `0.5329`; penalized upper risk changes from `0.7981` to `0.7127` | The namespace result is visible at practical route depths |
| Released-axis attribution | Policy-only versus namespace dense: recall `+0.0265`, feasible rate `+0.0576`, penalized risk `-0.1467`; lifecycle-only is harmful | Released policy drives the incremental governance benefit; the coarse lifecycle rule is not deployable |
| Natural text verification, 72 analysis cases | GPT-5.6 and Gemini text filters lose recall and increase penalized risk | Text-only verification does not recover released-field headroom selectively in the tested diagnostic |
| Controlled verification | Stable-control overflip is `0.2031`, `0.2969`, and `0.2656` for GPT-5.6, Gemini, and DeepSeek | Following focal rule changes is not sufficient for selective enforcement |
| Paired exposure | Selectivity gaps are `0.906`, `0.812`, `0.656`, and separately `0.844` for Claude; DeepSeek retains a positive inadmissible exposure effect | Once content enters the prompt, reader restraint is not a reliable enforcement boundary |
| GateMem observational audit | Exposure predicts disclosure; stricter exposure also lowers bounded utility and raises over-refusal | The safety-utility trade-off is real, but the estimate is non-causal and from a separate population |

These evidence blocks motivate the end-to-end evaluation. They do not establish that a
natural support constraint improves downstream answers and protected disclosure on
the same queries.

## 3. The remaining scientific question

On the same frozen RHELM/MemOps population, compare:

`retrieval -> support/filter -> prompt exposure -> reader answer -> score`

The primary estimands are paired route differences for:

- answer utility;
- over-refusal;
- protected disclosure where source-defined targets exist;
- wrong-namespace, policy-disallowed, and lifecycle-incompatible exposure; and
- prompt width and route cost.

RHELM has answer gold but no equivalent protected target. Protected disclosure is
therefore a MemOps-only estimand with explicit coverage. It must never be imputed for
RHELM.

## 4. Current incomplete execution

The v2 verifier is complete and reusable after the existing hash-bound compatibility
checks. The reader and judge stages are incomplete:

| Stage/provider | Saved successes | Saved failures | Comparison eligible |
| --- | ---: | ---: | --- |
| OpenAI verifier | 3,767 / 3,767 | 0 | Yes, as a frozen upstream bundle |
| Claude reader | 1,463 / 7,250 unique prompts | 2 | No |
| Gemini reader | 3,233 / 7,250 unique prompts | 11 | No |
| DeepSeek reader | 2,741 / 7,250 unique prompts | 5 | No |
| OpenAI reader | 0 / 7,250 unique prompts | 0 | No |
| Judge | Not started | - | No |

All partial reader responses remain private engineering checkpoints. Do not score
prefixes, choose a sample based on their outcomes, or pool them with a new protocol.
Archived v1 and overlapping-process runs remain separate and invalid for claims.

## 5. Sunk-cost audit

The local response envelopes contain `$551.943668` of protocol-price estimates for
successful calls in this natural end-to-end effort. A few invalid model completions
lack validated usage, so provider dashboards may differ by a small amount.

| Cost block | Recorded cost (USD) |
| --- | ---: |
| Current valid verifier plus partial v2 readers | 365.14 |
| Archived overlapping verifier and obsolete v1 reader runs | 186.80 |
| Total recorded successful calls | 551.94 |

Provider totals are OpenAI `$465.90`, Anthropic `$52.41`, Gemini `$21.47`, and
DeepSeek `$12.17`. No judge cost has been incurred.

For scale, the checked-in usage tables for the completed natural text-verification,
controlled verification, original paired-exposure, and Claude replication packages
sum to approximately `$13.56`. The incomplete natural end-to-end effort has therefore
already cost about 41 times those four completed provider-backed diagnostics combined.
This comparison excludes historical GateMem costs that are not represented by a
public provider-usage table in this repository.

At the same observed token volumes, repricing all calls with current low-cost family
models would be approximately `$110.28`, an 80% reduction. Removing obsolete runs
and exact-deduplicating the verifier reduces the comparable completed workload to
approximately `$49.29`. These are retrospective estimates, not a forecast of the
unfinished judge stage.

Candidate low-cost prices used for planning:

- OpenAI GPT-5.6 Luna: `$1` input / `$6` output per million tokens;
- Anthropic Claude Haiku 4.5: `$1` input / `$5` output per million tokens;
- Gemini 3.5 Flash-Lite: `$0.30` input / `$2.50` output per million tokens; and
- DeepSeek V4 Flash: `$0.14` cache-miss input / `$0.28` output per million tokens.

Official price sources checked on 2026-08-08:

- <https://openai.com/api/pricing/>
- <https://www.anthropic.com/claude/haiku>
- <https://ai.google.dev/gemini-api/docs/latest-model>
- <https://api-docs.deepseek.com/quick_start/pricing>

Pricing must be rechecked immediately before execution. A model substitution creates
a new experimental arm and must not be represented as continuation of an old arm.
The currently configured OpenAI and Gemini projects returned explicit depleted-credit
errors during content-free diagnostics. Do not purchase credits or replace keys until
Stage 0 passes and the two-reader amendment is committed.

## 6. Stage 0: zero-call eligibility audit

Before any new provider call:

1. Verify parseability and answer-gold presence for all 3,767 cases.
2. Confirm the protected-target denominator and target type for the 2,548 eligible
   MemOps cases.
3. Inspect a frozen, source-stratified 60-case content sample for answer-gold
   adjudicability and protected-target meaning. This is a source-quality audit, not a
   new population-wide annotation study.
4. Enumerate route identity and disagreement for all five frozen arms without reader
   outputs.
5. Freeze a sampling manifest before inspecting any partial reader answer.
6. Verify that the completed Sol verifier can be reused byte-for-byte; do not call it
   again.

Stage 0 gates:

- at least 95% of sampled answer gold is adjudicable;
- at least 90% of sampled protected targets denote meaningful disclosure content;
- all inclusion, exclusion, and unresolved rules are executable without answer gold;
- no sample membership depends on a reader response; and
- all route and sample artifacts are hash-bound and reproducible.

If a gate fails, stop. Fix the evaluation target rather than buying more generations.

Estimated API cost: `$0`.

Estimated author audit time: 2-4 hours for the frozen 60-case source-quality check.
This experiment does not require HPC: routes, embeddings, and verifier outputs are
already materialized, while generation is provider-hosted and scoring is local.

## 7. Stage 1: complete the Gemini and DeepSeek panel

Freeze a new two-reader amendment before any call. It retains the same 3,767 queries,
five route arms, reader prompt, output schema, 2,048-token limit, and exact model IDs:

- Gemini `gemini-3.6-flash`; and
- DeepSeek `deepseek-v4-pro`.

Import an existing success only when provider, model, payload, request body, prompt,
schema, parser, and response hashes all match. The amended protocol changes panel
membership and comparison eligibility; it does not reinterpret a response or change
either reader's request.

Current reusable progress:

| Reader | Complete | Remaining | Expected remaining cost |
| --- | ---: | ---: | ---: |
| Gemini 3.6 Flash | 3,233 / 7,250 | 4,017 | $24.46 |
| DeepSeek V4 Pro | 2,741 / 7,250 | 4,509 | $13.90 |
| Combined | 5,974 / 14,500 | 8,526 | $38.36 |

Quota, credential, or transport failures that produced no model answer may resume the
exact frozen request. Historical malformed or schema-invalid attempts remain in the imported audit
trail. In the v2 checkpoint, each of those request IDs already had a later accepted
response before this amendment; the v3 import preserves that response and the earlier
failure records separately. A new v3 model-contract failure receives exactly one
predeclared recovery attempt on the unchanged request. The output is never repaired;
a second model-contract failure for that request is terminal and freezes Stage 1 as
incomplete until a separately reviewed protocol amendment.

Stage 1 gates:

- all accepted imports pass exact compatibility checks;
- one single-writer process per provider;
- no new terminal model-output failure (historical failures remain reported);
- no prompt, route, model, token-cap, or parser drift; and
- no performance score is computed until both reader stages terminate under the
  amended completeness rule.

Gemini currently requires funded project credits. Do not top up or execute until the
amendment, tests, and content-free fixtures pass.

Expected incremental cost: `$38.36`.

Hard cap: `$45`.

## 8. Stage 2: deterministic triage and low-cost judging

### Stage 2A: zero-cost deterministic triage

After both readers terminate, compute paired source- and reader-specific outcomes for:

- evidence recall and feasible rate;
- typed prompt exposure and route width;
- normalized reference containment;
- literal protected-target disclosure where evaluable; and
- explicit refusal.

Stop before semantic judging if the constrained routes do not change the relevant
exposure surface, or if every constrained arm loses more than 0.05 deterministic
utility against global dense for both readers.

Estimated API cost: `$0`.

### Stage 2B: low-cost blinded judge

Run only if Stage 2A passes. Use a frozen Claude Haiku 4.5 judge, blinded to route arm
and reader identity, on exact-deduplicated answer/reference payloads. Rejudge a frozen
200-output sample with one predeclared strong judge and report agreement and
disagreements without repairing reader outputs.

The two-reader result is supportive only if both readers satisfy all of the following
for at least one primary constrained arm versus global dense:

1. judged utility point estimate no worse than `-0.02` and neither reader worse than
   `-0.05`;
2. over-refusal increase no greater than `+0.03`;
3. wrong-namespace exposure decreases in the expected direction;
4. MemOps protected-disclosure point estimate decreases where targets are evaluable;
5. each reader reaches at least 0.60 judged answer accuracy on at least one primary
   arm; and
6. cheap-versus-strong judge agreement is at least 0.85 on the frozen audit sample.

These are continuation gates, not released claims. Confidence intervals and negative
results remain reportable.

Expected incremental cost: `$20-$45`.

Hard cap: `$60`.

## 9. Stage 3: optional Luna and Haiku reader replication

Run only after the two-reader result is supportive. Add two new reader arms from
scratch:

- OpenAI GPT-5.6 Luna; and
- Anthropic Claude Haiku 4.5.

These are not continuations of GPT-5.6 Sol or Claude Opus 5. Their outputs must remain
model-specific and cannot be pooled with Gemini or DeepSeek. Restrict paid replication
to the three primary routes: global dense, namespace dense, and namespace plus
released-policy gate. Retain the two diagnostic routes from the Gemini/DeepSeek stage.

Estimated reader cost is `$28-$35` for Luna and `$52-$60` for Haiku. Semantic judging,
fixtures, and a strong-judge audit add an estimated `$20-$45`.

Expected incremental cost: `$100-$140`.

Hard cap: `$160`.

## 10. Total forward budget

| Decision path | Additional expected spend | Additional hard cap |
| --- | ---: | ---: |
| Stage 0 fails | $0 | $0 |
| Complete Gemini and DeepSeek only | $38.36 | $45 |
| Complete and judge the two-reader panel | $58-$83 | $105 |
| Add Luna and Haiku after supportive results | $158-$223 | $265 |

The `$551.94` already incurred is sunk and excluded from these forward caps. The
current authorization boundary is Stage 1's `$45` cap. Stage 2B and Stage 3 remain
locked behind their preceding evidence gates.

## 11. Execution safeguards

- New protocol ID, committed sample manifest, prompt hashes, model IDs, prices, and
  per-stage caps before any call.
- One single-writer process per provider stage.
- Atomic checkpoint after every accepted response.
- Exact prompt deduplication before submission.
- Content-free provider fixture before benchmark execution.
- Reader fixture caps of `$0.02` for Gemini and `$0.01` for DeepSeek, charged
  against the same provider-specific incremental caps as benchmark execution.
- No semantic repair, response repair, selective rerun, prefix scoring, or model
  substitution.
- Provider failures stop new submissions; successful checkpoints remain reusable.
- Conservative in-flight reservation before each call and an automatic stop before a
  provider-specific incremental hard cap can be exceeded.
- Partial bundles are never comparison eligible.
- Existing v2 reader checkpoints remain frozen and separate from the new protocol.

## 12. Decision value

A supportive result would close the evaluation's main empirical gap: on one natural
population, pre-prompt constraints would reduce inadmissible exposure or disclosure
without sacrificing answer utility. A mixed result would still localize the boundary
between retrieval improvement and downstream behavior. A negative result would
prevent an unsupported end-to-end claim and leave the released evidence as separate
retrieval and mechanism studies.

No outcome changes the already frozen claims unless the new protocol completes and a
separate claim-contract amendment is reviewed.
