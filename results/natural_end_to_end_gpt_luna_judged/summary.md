# GPT-5.6 Luna natural route-to-reader replication

This non-official replication evaluates the frozen 1,523-case sample under global dense, namespace dense, and namespace plus released-policy gating. GPT estimates are reported separately from Gemini and DeepSeek.

## Paired findings

- Namespace dense versus global dense: answer accuracy +0.0659 [+0.0390, +0.0961]; penalized admissibility upper risk -0.0847 [-0.1068, -0.0622]; over-refusal -0.0381 [-0.0589, -0.0148].
- Released-policy gate versus namespace dense: answer accuracy -0.0055 [-0.0178, +0.0064]; penalized admissibility upper risk -0.0192 [-0.0230, -0.0156].
- Namespace effects on protected disclosure and stale disclosure have 95% intervals that include zero; this replication does not establish a general disclosure reduction.
- Confidence intervals use 10,000 paired namespace-group bootstrap replicates with equal-source macro aggregation.

This is a prespecified robustness replication, not an official RHELM or MemOps submission. It does not rerun the text-verifier or released-field-oracle diagnostic routes and makes no pooled cross-reader claim.
