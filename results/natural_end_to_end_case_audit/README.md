# Natural End-to-End Case Audit

This package exposes the finest safe public derivative of the frozen natural
route-to-reader evaluation. `case_scores.csv` contains one row per
reader--case--route assignment. Raw case and namespace identifiers are replaced by
domain-separated SHA-256 tokens. The file contains route scores, parsed reader action,
judge labels, and SHA-256 bindings to the private reader and judge records; it contains
no query, memory, reference-answer, prompt, model-answer, or judge-rationale text.

`source_specific_deltas.csv` reports paired source-specific effects with 10,000
namespace-group bootstrap replicates. `case_weighted_sensitivity.csv` is a post-hoc
sensitivity analysis that gives every evaluable case equal weight across sources;
its bootstrap resamples namespace groups within source. `population_summary.csv`
reports anchor-count and protected-target coverage without source text.

Run `python -m scripts.publish_natural_case_audit verify` to reconstruct the checked-in
aggregate tables and paired intervals from `case_scores.csv`. This verification path
does not read private files or call a provider. The legacy aggregate field
`over_refusal` is represented here by the accurate name `non_answer`: it equals one
exactly when the reader action is not `answer`; it is not a gold-action-aware
over-refusal measure.

The original provider records and benchmark payloads are intentionally absent.
RHELM and MemOps source text must be obtained from the pinned upstream owners, and the
current license audit does not authorize redistribution of incorporated benchmark
text. Consequently this package supports result and bootstrap regeneration, plus
hash verification against a separately held raw bundle; it does not independently
replay the historical payload-to-score boundary.
