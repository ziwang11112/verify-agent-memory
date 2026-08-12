# Counterfactual Error Taxonomy

This is a posthoc descriptive analysis of the frozen controlled diagnostic.
It changes no prompt, prediction, threshold, gate, or comparison eligibility.

| Model | Stable-admissible false deny | Stable-inadmissible false admit | Stable overflip | Worst overflip axis |
| --- | ---: | ---: | ---: | --- |
| DeepSeek/deepseek-v4-pro | 0.5938 | 0.0156 | 0.2656 | lifecycle_intent (0.4375) |
| Gemini/gemini-3.6-flash | 0.5312 | 0.0000 | 0.2969 | lifecycle_intent (0.5000) |
| OpenAI/gpt-5.6-sol | 0.2656 | 0.0312 | 0.2031 | lifecycle_intent (0.4375) |

`false deny` means an explicitly stable-admissible memory was called
inadmissible. `false admit` means an explicitly stable-inadmissible memory
was called admissible. Overflip counts any change in the argmax label between
the two query conditions, including transitions through `unknown`.

These rates diagnose selectivity after the primary result was frozen. They are
not a new preregistered endpoint and must not be used to tune or rerun providers.

## Mechanism Read

The errors are strongly asymmetric. Stable-inadmissible memories are rarely
false-admitted, while stable-admissible memories are often false-denied or
sent to `unknown`. This is a verifier-level over-refusal pattern rather than
evidence of broad failure to detect explicit violations.

| Model | Lifecycle allow accuracy | Lifecycle block accuracy |
| --- | ---: | ---: |
| DeepSeek/deepseek-v4-pro | 0.5625 | 1.0000 |
| Gemini/gemini-3.6-flash | 0.5000 | 1.0000 |
| OpenAI/gpt-5.6-sol | 0.5625 | 1.0000 |

All three models have their largest stable-control overflip on lifecycle
intent. Their lifecycle control accuracy is lower on the history-oriented
`allow` side and reaches 1.0 on the current-state `block` side. Because the
prompt explicitly said not to score topical relevance, this pattern is
consistent with a relevance--admissibility conflation: a current or generally
permitted memory is treated as inadmissible when it is less directly useful to
the changed query. This is a posthoc interpretation of controlled cases, not a
causal diagnosis of model reasoning.
