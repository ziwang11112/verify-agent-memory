# Counterfactual Admissibility Diagnostic

This is a controlled public-development diagnostic, not an official benchmark result.
Candidate pools are fixed within each pair; only the query condition changes.

| Model | Strict focal pair | Direction | Stable overflip | Candidate accuracy |
| --- | ---: | ---: | ---: | ---: |
| released_oracle | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| no_verifier_keep_all | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| OpenAI/gpt-5.6-sol | 1.0000 | 1.0000 | 0.2031 | 0.8490 |
| DeepSeek/deepseek-v4-pro | 0.7188 | 0.8750 | 0.2656 | 0.6927 |
| Gemini/gemini-3.6-flash | 1.0000 | 1.0000 | 0.2969 | 0.8021 |

The frozen supportive pattern requires at least 0.80 strict focal-pair consistency,
at least 0.75 on every axis, stable-control overflip at most 0.05, and a
positive mean directional margin. Passing would establish explicit-condition rule
application only; it would not establish latent metadata inference or an E6/M2 gate.

## Interpretation

No comparison-eligible provider met the complete supportive pattern. GPT-5.6 Sol and
Gemini 3.6 Flash classified the focal flip correctly in all 32 pairs, with scenario-
bootstrap 95% intervals of `[1.000, 1.000]`. DeepSeek-V4-Pro reached `0.7188`
(`[0.5938, 0.8438]`) and fell to `0.5000` on both principal scope and lifecycle
intent.

Perfect focal performance did not imply selective verification. Stable controls were
supposed to retain the same decision across the two query conditions. Their overflip
rates were `0.2031` for GPT-5.6 Sol (`[0.1094, 0.3125]`), `0.2656` for
DeepSeek-V4-Pro (`[0.1875, 0.3438]`), and `0.2969` for Gemini 3.6 Flash
(`[0.2188, 0.3750]`), all above the frozen `0.05` ceiling. The controlled result is
therefore mixed: strong models follow the intended focal condition, but they also
propagate query changes to memories whose admissibility should remain invariant.

Claude Sonnet 5 passed the one-case provider fixture, then returned an invalid
candidate count during full execution. Its single valid prefix case is retained for
audit but excluded from every performance table; there was no repair, rerun, or prefix
score. Three providers completed all `64/64` cases.

Known fixture and execution cost across all four providers was approximately
`$0.7087`; one failed Claude call may be unaccounted because the invalid response did
not expose validated usage. This remains a controlled public-development diagnostic,
not an official benchmark result or a claim that an LLM verifier can replace trusted
namespace, policy, or lifecycle controls.
