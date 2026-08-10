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
- Completed cap-accounted cost: USD 1.85806

The experiment reuses the frozen 16 scenarios, 192 paired units, prompts, candidate
order, disclosure markers, and deterministic scorer. Claude is reported as a fourth
reader-specific estimate. The earlier three-reader estimates remain unchanged.

## Why this arm

The existing reader panel already includes OpenAI's flagship Sol tier, Google's
production Flash tier, and DeepSeek's Pro tier. Claude Opus 5 adds a high-capability
Anthropic arm and directly tests whether the prompt-boundary result holds
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

## Completed result

The hash-bound execution completed all 384 scored requests plus the compatibility
fixture with one attempt per request. The local deterministic scorer produced 192
paired units:

| Quantity | Estimate | 95% scenario-bootstrap CI |
| --- | ---: | ---: |
| Relevant and admissible exposure effect | 0.9688 | [0.9062, 1.0000] |
| Relevant and inadmissible exposure effect | 0.1250 | [0.0000, 0.2812] |
| Irrelevant and admissible exposure effect | 0.2656 | [0.1406, 0.4219] |
| Irrelevant and inadmissible exposure effect | 0.0000 | [0.0000, 0.0000] |
| Selectivity gap | 0.8438 | [0.6875, 0.9688] |

The positive selectivity-gap interval reproduces the qualitative result for a fourth
provider. The relevant-inadmissible point estimate remains nonzero, although its
interval includes zero. This is a controlled prompt-level reader replication, not an
official benchmark result, a natural-prevalence estimate, or a model leaderboard.

The content-free publication is in
`results/claude_opus5_exposure_replication/`. Raw prompts and responses remain under
the ignored runtime directory and are not published.

## Local verification

```powershell
python -m scripts.run_claude_opus5_exposure_replication validate
python -m pytest tests/test_claude_opus5_replication.py -q
python -m scripts.publish_claude_opus5_exposure_results verify
```

These commands do not call Anthropic. The completed provider execution was externally
unlocked against exact commit `938320909b4c2d13e286987dc4297a7cb6ef73a7`, execution
protocol SHA-256
`a01c17e1bff6dc00f7ba631c24dfc0038334805f06a9d4a0ee1583bff7efc11e`, and model
`claude-opus-5`. The checked-in result manifest retains the response-bundle, scoring,
and execution receipt hashes without publishing their contents.
