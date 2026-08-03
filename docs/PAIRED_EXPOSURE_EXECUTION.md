# Paired Exposure Provider Execution

This document describes the separately gated provider layer for the paired exposure
intervention. It is an execution contract, not a result report. The checked-in status
is `implementation_only_paid_execution_not_authorized`; no credential read or provider
call is authorized by the repository state alone.

## Frozen Reader Panel

| Provider | Model and controls | Cap-accounting price per 1M tokens | Provider cap |
| --- | --- | ---: | ---: |
| OpenAI | `gpt-5.6-sol`, reasoning effort `none` | $5.00 input / $30.00 output | $5.25 |
| Gemini | `gemini-3.6-flash`, thinking level `minimal` | $1.50 input / $7.50 output | $1.50 |
| DeepSeek | `deepseek-v4-pro`, thinking disabled | $0.87 input / $1.74 output | $0.70 |

The DeepSeek accounting prices are twice the published regular prices to cover the
announced peak-price guard conservatively. The total hard cap is $8.50. Models are
reported separately and are never pooled.

Pricing and model availability were checked on 2026-08-02 against the official
[OpenAI GPT-5.6 Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[Gemini 3.6 Flash documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash),
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing), and
[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/). Runtime receipts
record actual provider usage, while cap enforcement uses the conservative rates above.

## Call And Cost Bounds

Each provider receives 384 scored requests plus one compatibility fixture, for a
maximum of 385 calls. The scored request bundle contains 514,818 UTF-8 input bytes;
the largest request contains 1,528 bytes. Budget prechecks conservatively treat one
UTF-8 byte as one input token and reserve the full 192-token output allowance for
every call. The fixture answer is validated and discarded; it is not part of scoring.

The reader panel uses no tools, search, state, response reuse, output repair, semantic
retry, or LLM judge. OpenAI output usage includes reasoning tokens. The exposure
Gemini adapter charges `totalTokenCount - promptTokenCount`, so hidden thinking tokens
cannot disappear from budget accounting.

## Resume And Failure Contract

Every successful scored response is appended and `fsync`ed immediately. An ordinary
process interruption leaves a valid prefix and the next invocation resumes at the
first missing request. Existing rows are rebound to their exact request order,
payload hash, model, base protocol, and execution protocol before continuation.

Provider, transport, or output-contract failures create a frozen failure marker.
Automatic rerun, selective rerun, and output repair are prohibited. Transport retries
are fixed at zero. Missing local credentials fail before a provider attempt and do
not create a provider-failure marker.

Scoring requires all 384 responses for one provider. It publishes only content-free
pair scores, cell aggregates, bootstrap intervals, and hash manifests under `tmp/`;
raw answers are not copied into tracked results.

## Authorization And Commands

Paid commands require an external JSON unlock matching the exact Git commit,
execution-protocol hash, base-protocol hash, three model IDs, owner, and $8.50 cap.
Dirty files under `experiments/`, `scripts/`, or `src/` invalidate the unlock. Local
paper edits outside those contract-bearing paths do not alter the experiment.

The credential-free sequence is:

```powershell
python -m scripts.run_counterfactual_exposure_execution validate
python -m scripts.run_counterfactual_exposure_execution unlock-template --owner "zi wang"
```

Only after explicit approval and installation of the exact unlock may the operator
run, separately for `OpenAI`, `Gemini`, and `DeepSeek`:

```powershell
python -m scripts.run_counterfactual_exposure_execution preflight
python -m scripts.run_counterfactual_exposure_execution fixture --provider OpenAI
python -m scripts.run_counterfactual_exposure_execution execute --provider OpenAI
python -m scripts.run_counterfactual_exposure_execution score --provider OpenAI
```

Replacing `OpenAI` with the other provider names runs their independent bundles. An
approved execution remains a controlled public-development diagnostic, not an
official benchmark or full-paper decision.
