# Reproducibility Guide

This guide separates checks that are fully reproducible from this checkout from
historical executions that require external public data, model artifacts, or provider
access. All commands are run from the repository root.

## 1. Environment

Requirements:

- Python 3.11 or newer;
- Git 2.30 or newer;
- `uv` for the locked environment; and
- no GPU or API key for the default verification path.

Install the test and plotting dependencies:

```powershell
uv sync --extra dev --extra plots
```

## 2. Verify the Released Package

This tier is deterministic, offline after dependency installation, and makes no model
or provider call:

```powershell
uv run --extra dev --extra plots python -m pytest
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
```

The checks validate schemas, formulas, frozen settings, source hashes, normalized
evidence, result manifests, strict JSON/CSV parsing, secret exclusions, and the absence
of release-excluded private-data files.

## 3. Run the Local Retrieval Smoke

The synthetic fixture exercises the same normalized schema as the natural evaluation:

```powershell
uv run --extra dev python -m scripts.run_retrieval_experiment validate `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json

uv run --extra dev python -m scripts.run_retrieval_experiment run `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/retrieval_routes.jsonl

uv run --extra dev python -m scripts.run_retrieval_experiment select `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/selected_settings.jsonl
```

Run deterministic metadata and fixed-budget diagnostics on the same fixture:

```powershell
uv run --extra dev python -m scripts.run_metadata_robustness `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --retrieval-protocol experiments/metadata_robustness_retrieval_protocol.json `
  --robustness-protocol experiments/metadata_robustness_protocol.json `
  --output tmp/metadata_curve.jsonl `
  --break-even-output tmp/metadata_break_even.jsonl

uv run --extra dev python -m scripts.run_top_k_pareto `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --retrieval-protocol experiments/frozen_natural_protocol.json `
  --pareto-protocol experiments/top_k_pareto_protocol.json `
  --output tmp/top_k_pareto.jsonl
```

These are implementation checks, not benchmark estimates.

## 4. Validate Controlled Inputs and Results

Materialize provider-neutral paired exposure requests without calling a provider:

```powershell
uv run --extra dev python -m scripts.run_counterfactual_exposure_intervention validate
uv run --extra dev python -m scripts.run_counterfactual_exposure_intervention requests `
  --model local-contract-smoke `
  --output tmp/counterfactual_exposure/requests.jsonl
```

Verify the checked-in derived result packages and regenerate their public plots:

```powershell
uv run --extra dev python -m scripts.publish_supplemental_results verify
uv run --extra dev python -m scripts.publish_counterfactual_exposure_results verify
uv run --extra dev python -m scripts.publish_claude_opus5_exposure_results verify
uv run --extra dev python -m scripts.publish_natural_case_audit verify
uv run --extra dev python -m pytest tests/test_posthoc_robustness_results.py
uv run --extra dev --extra plots python -m scripts.plot_counterfactual_selectivity_figure
uv run --extra dev --extra plots python -m scripts.plot_counterfactual_exposure_figure
```

Plot scripts consume checked-in score tables only. They do not read credentials or raw
provider responses.

The checked-in support-control and operating-curve bundles are fully hash-verifiable
from Git. Recomputing them from frozen rankings or provider predictions additionally
requires the excluded provenance archive:

```powershell
python -m scripts.run_frozen_natural_support_controls `
  --archive-root ../bomi-codex-starter `
  --output-dir tmp/support_controls

python -m scripts.run_frozen_posthoc_robustness `
  --cases tmp/inferred_admissibility/cases.jsonl `
  --response-dir tmp/inferred_admissibility_canonical/primary_first_committed `
  --output-dir tmp/posthoc_robustness
```

Both commands are zero-call post-hoc analyses. The first reads frozen embeddings and
route bundles; the second reads frozen structured verifier responses. Neither command
reads credentials or contacts a provider.

## 5. Acquire Exact Public Sources

Validate the pinned source registry without network access:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources validate
```

Fetch and verify GateMem, RHELM, and MemOps from their owners:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources fetch
uv run --extra dev python -m scripts.fetch_public_sources verify
```

Checkouts are placed in ignored `data/raw/upstream/` directories. The tool verifies
both exact commit and Git tree hashes and refuses to replace a non-Git path or use a
mismatched remote.

## 6. Full-Execution Boundary

The repository supports three different reproducibility claims:

| Tier | Reproducible from this checkout? | Additional material |
| --- | --- | --- |
| Code behavior and synthetic smoke | Yes | None |
| Every checked-in aggregate/result/evidence hash | Yes | None |
| Natural case-level aggregation, equal-source results, and case-weighted sensitivity | Yes | Tokenized `results/natural_end_to_end_case_audit/case_scores.csv` |
| Raw natural-corpus and provider execution | No, not from Git alone | Upstream raw text, frozen embeddings/checkpoints, credentials, and response bundles |

The exact historical natural execution used 182,908 memories, 3,767 queries, 33,903
route rows, and 33,903 score rows. Its source revisions, config, population,
embedding, route, and score hashes are recorded in `PROVENANCE.md`. Embedding shards
would be several gigabytes, and provider responses include benchmark text; neither is
appropriate for ordinary Git storage.

The case-level audit exposes parsed numeric labels and SHA-256 bindings, so the public
package can recompute source-specific intervals, the post-hoc case-weighted
sensitivity, and every natural closure aggregate.
It cannot independently verify how a private provider response was converted into a
parsed score without the excluded response and benchmark payload. Provider execution
scripts remain available for audit, but a checked-in historical receipt is not
authorization to spend money or rerun a model. Complete-bundle, cost-cap, and
fail-closed requirements are enforced in code and tests.

## 7. Directory Contract

```text
src/           pure evaluation library
scripts/       local runners, importers, publishers, and audits
experiments/   frozen protocols, prompts, and constructed inputs
data/          upstream revision registry and redistribution policy
results/       content-free derived result packages
evidence/      normalized claim-bound evidence and hash manifests
claims/        machine-readable interpretation boundaries
tests/         unit, invariant, provenance, and end-to-end smoke tests
docs/          method and execution contracts
```

See `data/README.md`, `experiments/README.md`, and `results/README.md` for complete
inventories.
