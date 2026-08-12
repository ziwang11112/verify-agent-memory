# Natural Cross-Judge Robustness Audit

## Why this audit exists

The natural route-to-reader closure reports DeepSeek V4 Pro, Gemini 3.6 Flash,
and GPT-5.6 Luna separately, but all three readers were scored by the same blinded
Claude Haiku 4.5 judge. The cost-aware plan already called for a 200-output check
with a stronger judge. No sample manifest or strong-judge model was frozen before
the reader results were available, so this audit must be described as **post-hoc but
outcome-independent**, not as an independently preregistered replication.

## Frozen design

- Judge: `gpt-5-pro-2025-10-06`, OpenAI Responses API, `high` reasoning.
- Prompt and JSON schema: byte-identical to the existing blinded semantic judge.
- Sample: 200 exact-deduplicated judge payloads.
- Panel: three readers, RHELM and MemOps, and the three common primary routes:
  `global_dense`, `namespace_dense`, and `namespace_policy_gate`.
- Selection: near-equal quotas across the 18 reader/source/route strata, followed
  by SHA-256 ranking under a fixed seed. Selection uses IDs and strata only. It
  never reads answer text, reference answers, Claude labels, or reader outcomes.
- Duplicate handling: a byte-identical judge payload is called once even if it is
  associated with multiple readers or routes.
- Primary endpoint: exact agreement on `answer_correct`.
- Existing continuation threshold: exact agreement at least `0.85`.
- Secondary diagnostics: Cohen's kappa, Gwet's AC1, quality-score agreement,
  directional confusion counts, and descriptive disclosure-label agreement.

The 0.85 threshold is a continuation criterion from the earlier cost-aware plan,
not a benchmark claim. Subgroup estimates are diagnostic because each individual
reader/source/route stratum contains only 11 or 12 sampled payloads.

## Cost and failure policy

The exact model price bound in the protocol is `$15/M` input tokens and `$120/M`
output tokens. Each benchmark call reserves `$0.096`, corresponding to 2,304 input
tokens and 512 output tokens. Together with the `$0.25` synthetic-fixture allowance,
the planned reservation is `$19.45` under a total incremental hard cap of `$20.00`.

There are no transport retries, output repairs, or selective reruns. Every accepted
response is checkpointed privately. A provider or model-contract failure freezes the
run and requires a separate recovery contract; the script will not automatically
reuse only favorable outputs.

## Privacy and reporting boundary

Benchmark questions, reference answers, reader outputs, judge responses, and judge
reasons remain in ignored `tmp/` storage. The tracked sample manifest contains only
IDs, sampling strata, and hashes. Public result files contain aggregate agreement,
cost, and integrity receipts only. Reader estimates remain separate, and this audit
does not create an official RHELM or MemOps result.

## Commands

```powershell
python scripts/run_natural_cross_judge_audit.py validate
python scripts/run_natural_cross_judge_audit.py fixture
python scripts/run_natural_cross_judge_audit.py execute
python scripts/run_natural_cross_judge_audit.py analyze
```

Provider calls are rejected unless every contract path is committed and clean.
