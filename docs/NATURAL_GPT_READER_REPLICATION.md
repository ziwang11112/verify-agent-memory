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
```

Provider calls require committed, clean contract files. Runtime questions, prompts,
responses, and query-level scores remain under the ignored
`tmp/natural_end_to_end/gpt_luna_replication_runtime/` directory.

This is a non-official robustness replication, not an official RHELM or MemOps result.
