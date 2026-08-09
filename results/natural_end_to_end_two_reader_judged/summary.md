# Cost-aware natural end-to-end route-to-reader evaluation

The semantic judge covers all 523 RHELM cases and a frozen, outcome-independent simple random sample of 1,000/3,244 MemOps cases. Every selected case is paired across both readers and all five routes; all 80 MemOps namespace groups are represented. Estimates describe this prespecified judge sample, not an official benchmark submission.

This is a non-official same-population diagnostic over frozen public-source 
RHELM and MemOps evaluation queries. Reader estimates are reported separately and 
are never pooled. RHELM has no source-defined protected target, so protected 
disclosure is reported for MemOps only.

| reader | arm | answer accuracy | over-refusal | evidence recall | admissibility risk | MemOps protected disclosure |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek | global_dense | 0.3719 | 0.5156 | 0.4321 | 0.7975 | 0.1055 |
| DeepSeek | namespace_dense | 0.4244 | 0.4671 | 0.5362 | 0.7128 | 0.0991 |
| DeepSeek | namespace_policy_gate | 0.4309 | 0.4721 | 0.5362 | 0.6936 | 0.1055 |
| DeepSeek | namespace_text_verifier | 0.4344 | 0.4622 | 0.5184 | 0.7195 | 0.0966 |
| DeepSeek | released_field_oracle | 0.4294 | 0.4736 | 0.5258 | 0.6970 | 0.1004 |
| Gemini | global_dense | 0.4566 | 0.3489 | 0.4321 | 0.7975 | 0.1004 |
| Gemini | namespace_dense | 0.5243 | 0.3084 | 0.5362 | 0.7128 | 0.1131 |
| Gemini | namespace_policy_gate | 0.5198 | 0.3164 | 0.5362 | 0.6936 | 0.1233 |
| Gemini | namespace_text_verifier | 0.5094 | 0.3169 | 0.5184 | 0.7195 | 0.1144 |
| Gemini | released_field_oracle | 0.5178 | 0.3169 | 0.5258 | 0.6970 | 0.1131 |

Interpretation must use `paired_deltas.csv`; point estimates alone do not establish 
an improvement. The released-field oracle is an upper bound, not a deployable method. 
The text verifier uses a threshold selected on public development data and is not 
retuned here. This experiment is not an official RHELM or MemOps benchmark result.

## Paired findings

- **Namespace support restriction is the robust positive result.** Relative to global dense retrieval, namespace dense raises answer accuracy for DeepSeek by +0.0525 [+0.0203, +0.0837] and for Gemini by +0.0676 [+0.0467, +0.0902]. It also raises evidence recall by +0.1040 [+0.0835, +0.1243], lowers penalized admissibility risk by -0.0847 [-0.1067, -0.0628], and lowers over-refusal for both readers.
- **Additional deletion gates reduce route risk but do not establish a utility gain.** The released-policy gate lowers risk relative to namespace dense by -0.0192 [-0.0229, -0.0157], while answer-accuracy intervals include zero for DeepSeek (+0.0065 [-0.0089, +0.0220]) and Gemini (-0.0045 [-0.0200, +0.0114]).
- **The text-only verifier is not supported as the main route.** Relative to namespace dense it increases penalized risk by +0.0068 [+0.0024, +0.0115]; Gemini answer accuracy falls by -0.0149 [-0.0283, -0.0011], while the DeepSeek interval includes zero.
- **Disclosure effects are reader-specific and not a general safety win.** Namespace-dense protected-disclosure intervals include zero for both readers. The policy gate's Gemini stale-disclosure delta is +0.0200 [+0.0030, +0.0385], whereas the DeepSeek interval includes zero; this axis should not be pooled or described as uniformly improved.

Taken together, the same-population result supports trusted namespace restriction as the main deployable intervention. It does not show that progressively stricter policy, text-verifier, or oracle deletion monotonically improves downstream answer quality or disclosure. The experiment remains a prespecified non-official sample, and all reader estimates remain separate.
