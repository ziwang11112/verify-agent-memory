# Result Packages

All tracked result packages are content-free derivatives. They may contain aggregate
metrics, pair-level binary/numeric scores, bootstrap intervals, model/token/cost
receipts, and hashes. They do not contain benchmark conversations, reference answers,
reader prompts, raw provider responses, embeddings, or credentials.

| Directory | Evaluation | Main artifacts |
| --- | --- | --- |
| `supplemental_natural/` | Natural retrieval, top-k Pareto, released-field attribution, metadata break-even | CSV/JSON summaries and manifests |
| `inferred_admissibility/` | Natural-development text verifier | Classification, route, threshold, sensitivity, usage, and manifest files |
| `counterfactual_admissibility/` | Controlled focal/stable verifier diagnostic | Pair scores, bootstrap intervals, error taxonomy, figures, manifest |
| `counterfactual_exposure/` | Original three-reader paired exposure intervention | Pair scores, cell metrics, bootstrap intervals, usage, figures, manifest |
| `claude_opus5_exposure_replication/` | Separately executed fourth-reader replication | Pair scores, cell metrics, bootstrap intervals, usage, manifest |
| `natural_end_to_end_two_reader_deterministic/` | Deterministic retrieval-side closure analysis | Main table, paired deltas, summary, manifest |
| `natural_end_to_end_two_reader_judged/` | DeepSeek/Gemini natural route-to-reader results | Main table, paired deltas, completion receipt, summary, manifest |
| `natural_end_to_end_gpt_luna_judged/` | Sequential GPT-reader replication | Main table, paired deltas, completion receipt, summary, manifest |
| `natural_cross_judge_audit/` | Outcome-independent 200-output alternate-judge audit | Agreement, confusion, subgroup, usage, summary, manifest |

## Verification

The normalized subset used for claim-level reporting is under `evidence/normalized/`.
Each evidence family has a manifest binding source paths, source SHA-256 values,
transformation code, row count, and normalized output hash.

Run all content and provenance checks without network or provider calls:

```powershell
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
uv run --extra dev python -m scripts.publish_supplemental_results verify
uv run --extra dev python -m scripts.publish_counterfactual_exposure_results verify
uv run --extra dev python -m scripts.publish_claude_opus5_exposure_results verify
```

Interpretation boundaries are maintained in `CLAIM_CONTRACT.md` and
`claims/claims.yaml`. Results from distinct readers, populations, or sequential
replications must not be pooled unless the corresponding contract defines that
estimand.
