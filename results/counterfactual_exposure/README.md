# Paired Counterfactual Exposure Intervention

This is a controlled, hash-bound diagnostic, not an official benchmark result. One
candidate is exposed or withheld while the query and all other candidates remain fixed.
Provider estimates are reported separately and are never pooled.

| Reader | Relevant + admissible ATE | Relevant + inadmissible ATE | Selectivity gap (95% CI) |
| --- | ---: | ---: | ---: |
| `gpt-5.6-sol` | 0.9062 | 0.0000 | 0.9062 [0.8125, 0.9688] |
| `gemini-3.6-flash` | 0.8125 | 0.0000 | 0.8125 [0.6875, 0.9375] |
| `deepseek-v4-pro` | 0.8125 | 0.1562 | 0.6562 [0.5000, 0.8125] |

## Interpretation boundary

Across all three frozen readers, exposure increased disclosure of relevant admissible
evidence substantially more than disclosure of relevant inadmissible evidence. DeepSeek
also showed a positive relevant-inadmissible exposure effect; the other two readers did
not show an aggregate incremental effect in that cell. This supports a controlled
prompt-level exposure claim only. It does not estimate natural-corpus prevalence, latent
metadata inference, production safety, or official benchmark performance.

The published bundle contains derived binary pair scores, aggregate metrics, uncertainty,
cost/latency summaries, and hashes. It contains no prompts, candidate text, literal target
markers, provider responses, answer text, API credentials, or private data.
