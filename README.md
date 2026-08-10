# Verify Agent Memory

An open-source evaluation toolkit for **retrieval admissibility** in long-term agent
memory. It tests whether retrieved evidence is not only relevant, but also valid for
the current principal, policy, intent, lifecycle state, and time.

```text
Stored -> Retrieved -> Exposed -> Disclosed
```

The package keeps these stages separate so a high-recall route cannot hide wrong-
namespace, policy-disallowed, stale, superseded, or answer-disclosed evidence.

## Included

- a pure Python evaluation library under `src/verify_agent_memory/`;
- frozen experiment protocols, prompt contracts, and constructed cases;
- deterministic retrieval, robustness, matched-recall, Pareto, verifier, exposure,
  reader, and judge-audit runners;
- all legally redistributable derived result packages;
- normalized evidence with source and transformation hashes;
- synthetic data for complete local execution-path testing;
- exact public-source Git revisions and a fetch/verification CLI; and
- unit, invariant, provenance, package-integrity, and end-to-end smoke tests.

The public package excludes private conversations, credentials, raw provider
responses, and embedding checkpoints.

## Quick Start

Requirements: Python 3.11+, Git, and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev --extra plots

uv run --extra dev python -m scripts.fetch_public_sources validate
uv run --extra dev python -m scripts.run_retrieval_experiment validate `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json
uv run --extra dev python -m scripts.run_retrieval_experiment run `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/retrieval_routes.jsonl
```

These commands make no provider call and require no API key or GPU.

## Verify Everything Released Here

```powershell
uv run --extra dev python -m pytest
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
uv run --extra dev python -m scripts.publish_supplemental_results verify
uv run --extra dev python -m scripts.publish_counterfactual_exposure_results verify
uv run --extra dev python -m scripts.publish_claude_opus5_exposure_results verify
uv run --extra dev python -m scripts.publish_natural_case_audit verify
```

The checks reject schema drift, non-finite data, changed hashes, incomplete result
families, unindexed evidence, release-excluded paths, and common secret/key shapes.

## Evaluation Surface

For a memory `m`, query `q`, policy context `p`, and time `t`:

```text
admissible(m, q, p, t)
  = scopeAllowed(m, q, p)
    AND policyAllowed(m, q, p)
    AND lifecycleCompatible(m, intent(q), t)

usable(m, q, p, t)
  = relevant(m, q)
    AND admissible(m, q, p, t)
```

Labels are three-valued: allowed, disallowed, or unresolved. Known violations are
excluded; unresolved labels remain unresolved and contribute to explicit coverage and
lower/upper risk bounds.

The main retrieval comparisons include global BM25/dense, hybrid and recency dense,
trusted namespace dense, released policy/lifecycle filters, threshold routing, and
cluster routing. All methods use the same candidate construction and development-
only setting-selection budget. Matched-recall evaluation reports recall, feasibility,
known admissibility risk, label coverage, conservative bounds, typed violations, and
route work.

Implementation details and equations are in
[`docs/EXPERIMENT_METHODS.md`](docs/EXPERIMENT_METHODS.md).

## Data

Three input classes are explicit:

| Class | Availability |
| --- | --- |
| Synthetic and constructed inputs | Checked in under `tests/fixtures/` and `experiments/` |
| GateMem, RHELM, and MemOps source repositories | Fetchable at exact commit and tree hashes with `scripts.fetch_public_sources` |
| Raw provider responses, benchmark-bearing prompts, embeddings, and checkpoints | Intentionally excluded; hashes and content-free receipts are retained |

Fetch and verify the public sources from their owners:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources fetch
uv run --extra dev python -m scripts.fetch_public_sources verify
```

See [`data/README.md`](data/README.md) for exact revisions, redistribution limits, and
the full execution boundary. Repository-level source licenses do not automatically
broaden rights in incorporated datasets.

## Results

Checked-in result families cover natural retrieval, fixed-budget and metadata
robustness, natural and controlled verification, paired exposure, three natural
reader bundles, a tokenized case-level audit, and an alternate-judge audit. Public
result files contain only derived scores, aggregates, intervals, figures, usage/cost
receipts, and hashes.

Representative frozen findings include:

- namespace-constrained support improves recall and admissibility risk relative to
  global dense on the natural evaluation;
- released policy metadata supplies most of the incremental governance gain, while
  the evaluated coarse lifecycle-only rule removes useful evidence;
- tested text-only verifiers over-deny stable evidence and do not recover the
  released-field utility-risk frontier;
- reader restraint is imperfect after relevant inadmissible evidence is exposed; and
- a 200-output alternate-judge audit reaches `0.865` answer-correctness agreement,
  with a weaker `0.833` sequential GPT-reader subgroup.

These are bounded evaluations, not an official benchmark leaderboard or a claim of a
new state-of-the-art memory index. Exact estimates and interpretation limits are in
[`results/README.md`](results/README.md), [`evidence/README.md`](evidence/README.md),
and [`CLAIM_CONTRACT.md`](CLAIM_CONTRACT.md).

## Repository Layout

```text
src/           evaluation library
scripts/       experiment runners, importers, publishers, and integrity checks
experiments/   frozen protocols, prompts, and constructed inputs
data/          public-source registry and redistribution policy
results/       content-free derived result packages
evidence/      normalized measurements and provenance manifests
claims/        machine-readable reporting boundaries
tests/         unit, invariant, and reproducibility tests
docs/          method and execution contracts
```

Start with [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full command matrix.

## Provenance and License

Every imported artifact is indexed in `SOURCE_ARTIFACTS.yaml`; upstream revisions,
execution identities, and checkpoint hashes are recorded in `PROVENANCE.md`.
Original code and documentation are MIT licensed. Third-party data and repositories
retain their own terms; see `THIRD_PARTY_NOTICES.md` and `docs/LICENSE_AUDIT.md`.
