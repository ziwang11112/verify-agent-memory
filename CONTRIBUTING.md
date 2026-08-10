# Contributing

Contributions that improve evaluation correctness, reproducibility, documentation, or
coverage are welcome.

## Development Setup

```powershell
uv sync --extra dev --extra plots
uv run --extra dev python -m pytest
```

## Pull Request Checklist

Before opening a pull request:

1. Keep runtime evaluation code pure and deterministic where possible.
2. Add tests for changed formulas, schemas, edge cases, or execution boundaries.
3. Do not tune settings on evaluation/test outcomes.
4. Do not commit raw benchmark text, provider responses, embeddings, checkpoints,
   credentials, `.env` files, or private user data.
5. Add a manifest and source hashes for every new result or evidence family.
6. Keep model, reader, source, and population estimates separate unless an explicit
   estimand defines aggregation.
7. Run the complete verification suite from `REPRODUCIBILITY.md`.

## Result Changes

A result change must include:

- the frozen protocol or configuration;
- the exact implementation revision;
- a content-free completion/provenance manifest;
- deterministic transformation code;
- normalized evidence where the result supports a claim; and
- an updated interpretation boundary in `claims/claims.yaml`.

Provider-backed results must also report model identifiers, request counts, token or
usage accounting, failure/retry behavior, and cost. A historical execution receipt is
not permission to rerun paid calls.

## Style

- Python 3.11+;
- Ruff formatting and linting;
- ASCII unless an existing source contract requires Unicode; and
- concise comments only where behavior is not self-explanatory.
