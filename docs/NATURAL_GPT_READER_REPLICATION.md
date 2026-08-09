# GPT-only natural reader replication

This staged replication adds one cost-oriented OpenAI reader to the completed natural
route-to-reader experiment. It does not rerun retrieval, the frozen text verifier,
Gemini, DeepSeek, or either diagnostic deletion route.

## Frozen scope

- Reader: `gpt-5.6-luna`, Responses API, reasoning effort `none`.
- Sample: the previously frozen 523 RHELM census plus the outcome-independent 1,000
  MemOps sample used by the two-reader semantic judge.
- Routes: `global_dense`, `namespace_dense`, and `namespace_policy_gate`.
- Pairing: every selected case is evaluated under every route.
- Reader outputs remain model-specific and are never pooled with Gemini or DeepSeek.

The 4,569 case-route assignments deduplicate to 3,157 byte-identical visible prompts.
The reader has a `$45` incremental hard cap, including its synthetic fixture. The
larger `$110` conservative envelope is used only to prove that the full request set is
admissible under worst-case token reservations; the incremental cap remains the
effective spending limit.

## Staging

Reader execution is completed first. Only after a complete reader bundle exists may a
separate judge protocol bind the exact GPT response set and exact judge request set.
The judge remains cross-provider and route-blinded. No partial reader bundle, prefix,
or selectively chosen output may be scored.

```powershell
python scripts/run_natural_gpt_reader_replication.py validate
python scripts/run_natural_gpt_reader_replication.py plan-reader
python scripts/run_natural_gpt_reader_replication.py fixture-reader
python scripts/run_natural_gpt_reader_replication.py execute-reader
python scripts/run_natural_gpt_reader_replication.py validate-judge
python scripts/run_natural_gpt_reader_replication.py plan-judge
python scripts/run_natural_gpt_reader_replication.py fixture-judge
python scripts/run_natural_gpt_reader_replication.py execute-judge
python scripts/run_natural_gpt_reader_replication.py score
```

Provider calls require committed, clean contract files. Runtime questions, prompts,
responses, and query-level scores remain under the ignored
`tmp/natural_end_to_end/gpt_luna_replication_runtime/` directory.

This is a non-official robustness replication, not an official RHELM or MemOps result.
Claude Haiku is used only as the same blinded cross-provider scorer used for the
Gemini/DeepSeek panel; it is not a reader arm. Its exact request set is frozen only
after the complete GPT response bundle exists.

## Completed result

The reader completed 3,157/3,157 unique requests with no failures for `$15.878671`.
The cross-provider judge completed 3,397/3,397 unique requests with no failures for
`$5.899861`. Combined incremental cost was `$21.778532`.

At equal-source macro aggregation, namespace dense versus global dense changes:

- judged answer accuracy by `+0.0659` (95% CI `[+0.0390, +0.0961]`);
- evidence recall by `+0.1040` (`[+0.0832, +0.1246]`);
- penalized admissibility upper risk by `-0.0847` (`[-0.1068, -0.0622]`); and
- over-refusal by `-0.0381` (`[-0.0589, -0.0148]`).

The released-policy gate reduces route risk by `-0.0192`
(`[-0.0230, -0.0156]`) relative to namespace dense, but its answer-accuracy change
is `-0.0055` (`[-0.0178, +0.0064]`). Protected- and stale-disclosure intervals
include zero. The GPT result therefore replicates the namespace utility-risk effect
without establishing a general disclosure improvement or an incremental utility gain
from the additional policy deletion gate.
