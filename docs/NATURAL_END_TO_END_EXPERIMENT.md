# Natural route-to-reader evaluation

This experiment tests the full observable path on one frozen natural population:

`retrieval -> admissibility filter -> prompt exposure -> reader answer -> blinded score`

It uses the 3,767 evaluation queries already routed in the pinned RHELM and MemOps
source snapshots. It is a non-official diagnostic, not an official benchmark result.

## Frozen arms

All arms expose at most 20 candidates. The three filtering arms delete candidates from
the same frozen namespace-dense top-20 list; they do not backfill, rerank, or use answer
gold.

| Arm | Candidate construction |
|---|---|
| `global_dense` | Frozen global dense top-20 |
| `namespace_dense` | Frozen trusted-namespace dense top-20 |
| `namespace_policy_gate` | Drop source-released policy violations |
| `namespace_text_verifier` | Drop candidates whose fixed dev-selected violation probability is at least 0.95 |
| `released_field_oracle` | Drop known released-field scope, policy, or lifecycle violations; retain unresolved items |

The reader sees the query, optional as-of time, candidate text, and visible record order.
It never sees arm names, hidden memory IDs, relevance labels, or admissibility gold. Exact
reader-visible prompts may be evaluated once and shared across arms only when their bytes
are identical. Reader outputs are capped at 2,048 provider tokens; the earlier 512-token
pilot was frozen unscored after a valid Claude response reached the cap.

The v2 protocol may reuse the completed v1 verifier bundle because the only substantive
execution change is the downstream reader output cap. The import is allowed only after
all 3,767 verifier payload hashes, OpenAI request-body hashes, current-schema parses, and
the response-set hash match exactly. Imported envelopes retain source-record hashes and
a private compatibility manifest; verifier response content is unchanged.

## Gold coverage

- Answer utility: 3,767/3,767 queries.
- Source-defined protected targets: 2,548/3,244 MemOps queries.
- RHELM protected disclosure: unevaluable, because RHELM does not release an equivalent
  answer-level protected target.

RHELM eligibility is restricted to `conversation_only` questions whose released
supporting references are all uniquely resolvable (`all_references_unique`). Of 1,305
RHELM QA pairs at the pinned source revision, 859 meet both filters; the frozen split
assigns 523 of those eligible questions to evaluation.

Protected disclosure is therefore reported with an explicit denominator and never
imputed for RHELM. Retrieval and prompt-exposure metrics cover both sources.

## Metrics

Retrieval metrics are evidence recall, feasible rate at target recall 0.8, penalized
admissibility upper risk, route width, and typed wrong-namespace, policy, lifecycle, and
unresolved exposure. Answer metrics are blinded semantic accuracy, answer quality,
non-answer rate, reference containment, source-defined protected disclosure, and stale
disclosure. The frozen aggregate CSVs retain the legacy field name `over_refusal`, but
its exact implementation is `reader_action != "answer"`; it is not a gold-action-aware
over-refusal label. Reader models are reported separately. Paired 95% intervals use
10,000 namespace-group bootstrap replicates.

## Execution

Raw public-source text, prompts, and provider responses stay under the ignored
`tmp/natural_end_to_end/` directory. The public case-audit package may include only
tokenized, content-free numeric scores and private-record hashes; it never includes
queries, memory text, reference answers, model answers, or judge rationales.

```powershell
python scripts/materialize_natural_end_to_end_cases.py
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py validate
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py plan-verifier
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py fixture --stage verifier
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py execute-verifier
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py plan-readers
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py fixture --stage reader --provider OpenAI
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py fixture --stage reader --provider Anthropic
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py fixture --stage reader --provider Gemini
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py fixture --stage reader --provider DeepSeek
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py execute-reader --provider OpenAI
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py execute-reader --provider Anthropic
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py execute-reader --provider Gemini
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py execute-reader --provider DeepSeek
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py plan-judge
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py fixture --stage judge
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py execute-judge
.\.venv\Scripts\python.exe scripts\run_natural_end_to_end_experiment.py score
```

Every successful call is checkpointed separately. A provider failure stops new
submissions after the small in-flight window drains; completed calls remain resumable.
Each provider stage holds an atomic single-writer lock for its full execution. A stale
lock fails closed and must be audited rather than removed automatically. No semantic
repair or outcome-selective rerun is performed.
