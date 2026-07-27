# Normalized Evidence

This directory contains content-free aggregate measurements for claims C2-C8. It
does not contain benchmark conversations, query text, memory text, model responses,
reviewer identity, embeddings, or per-query identifiers.

## Evidence Families

| File | Population boundary |
| --- | --- |
| `normalized/gatemem.csv` | Separate fixed-reader exposure/use associations and a clean-reader leakage-utility trade-off |
| `normalized/mechanism_smoke.csv` | Sixteen curated candidate-pool evaluation packets only |
| `normalized/natural_evaluation.csv` | Source-macro evaluation over 87 groups, 182,908 memories, and 3,767 queries |
| `normalized/human_agreement.csv` | Raw pre-adjudication agreement over 207 selected audit records |

Every CSV uses the same schema:

```text
claim_id,family,population,source,contrast,metric,estimate,
ci95_lower,ci95_upper,n,notes
```

Blank confidence bounds mean that the frozen source did not report an interval for
that row. They are not zeroes. Delta rows are method minus reference, as named in
`contrast`. Source-macro values weight the RHELM and MemOps source estimates equally;
they are not pooled-query estimates.

## Integrity

Each file has a JSON receipt in `manifests/` containing:

- its normalized SHA-256 and row count;
- every frozen source path and SHA-256 used to derive it;
- the read-only source snapshot identity; and
- the exact transformation-script SHA-256.

Run:

```powershell
uv run --extra dev python scripts/verify_evidence.py
```

The verifier rejects changed CSVs, changed transformation code, unindexed or changed
source hashes, path traversal, non-finite values, malformed intervals, retired method
labels, values or intervals that drift from `claims/claims.yaml`, and missing
claim-specific interpretation boundaries.
