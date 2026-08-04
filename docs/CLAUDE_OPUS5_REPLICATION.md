# Claude Opus 5 Reader Replication

This package adds a separate capability-oriented reader replication to the paired
counterfactual exposure experiment. It does not alter, replace, or pool the frozen
GPT-5.6 Sol, Gemini 3.6 Flash, and DeepSeek V4 Pro results.

## Exact model contract

- Provider: Anthropic
- Model ID: `claude-opus-5`
- API: Messages API
- Thinking: disabled, matching the low-deliberation design of the frozen panel
- Effort: medium
- Structured output: the existing `action` plus `answer` JSON contract
- Requests: 384 paired exposure requests plus one compatibility fixture
- Retries, output repair, selective reruns, judges, and pooling: disabled
- Paid hard cap: USD 5.00

The experiment reuses the frozen 16 scenarios, 192 paired units, prompts, candidate
order, disclosure markers, and deterministic scorer. Claude is reported as a fourth
reader-specific estimate. The earlier three-reader estimates remain unchanged.

## Why this arm

The existing reader panel already includes OpenAI's flagship Sol tier, Google's
production Flash tier, and DeepSeek's Pro tier. Claude Opus 5 adds a high-capability
Anthropic arm and directly tests whether the paper's prompt-boundary conclusion holds
for another provider. It is not described as Anthropic's most capable current model;
that designation belongs to Claude Fable 5 as of 2026-08-03. Opus 5 is selected here
because it permits disabled thinking at medium effort, which is closer to the frozen
panel's reasoning-none, thinking-minimal, and thinking-disabled controls. Adding
Fable 5 would be a different experiment with always-on adaptive thinking and is not
part of this replication.

This reader replication is more informative than replacing the retrieval encoder:
the encoder ranks candidates, while this experiment measures what happens after a
candidate is exposed to the reader.

The historical Claude Sonnet 5 verifier attempt is not reused. It failed a frozen
candidate-count output contract and was never scored. The Opus reader replication has
a much smaller two-field output schema and is a fresh, separately identified run.

## Zero-call preparation

```powershell
python -m scripts.run_claude_opus5_exposure_replication validate
python -m pytest tests/test_claude_opus5_replication.py -q
```

`validate` does not read credentials or call Anthropic. Paid commands require an
external unlock bound to the exact clean commit, protocol hash, model, owner, and
USD 5.00 cap.
