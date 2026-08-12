# Two-reader deterministic triage

This zero-call analysis reports Gemini and DeepSeek separately. It uses literal 
reference containment as a deterministic utility proxy; it is not semantic answer 
accuracy and is not an official RHELM or MemOps result.

| reader | arm | ref contains | over-refusal | recall | risk | wrong namespace |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek | global_dense | 0.0892 | 0.5217 | 0.4316 | 0.7995 | 0.3166 |
| DeepSeek | namespace_dense | 0.1044 | 0.4723 | 0.5329 | 0.7150 | 0.0000 |
| DeepSeek | namespace_policy_gate | 0.1041 | 0.4732 | 0.5329 | 0.6979 | 0.0000 |
| DeepSeek | namespace_text_verifier | 0.1045 | 0.4715 | 0.5134 | 0.7225 | 0.0000 |
| DeepSeek | released_field_oracle | 0.1029 | 0.4757 | 0.5230 | 0.7000 | 0.0000 |
| Gemini | global_dense | 0.1218 | 0.3459 | 0.4316 | 0.7995 | 0.3166 |
| Gemini | namespace_dense | 0.1355 | 0.3129 | 0.5329 | 0.7150 | 0.0000 |
| Gemini | namespace_policy_gate | 0.1352 | 0.3193 | 0.5329 | 0.6979 | 0.0000 |
| Gemini | namespace_text_verifier | 0.1329 | 0.3210 | 0.5134 | 0.7225 | 0.0000 |
| Gemini | released_field_oracle | 0.1349 | 0.3197 | 0.5230 | 0.7000 | 0.0000 |

Deterministic continuation gate: **PASS**.

The gate was frozen before outcome scoring. Passing only authorizes blinded semantic 
judging; it is not a paper claim and does not establish answer utility.
