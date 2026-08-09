# Anonymous Release

The submission artifact must not be a fork and must not retain the private
repository's Git history. It may contain only tracked, reviewable files from a clean
commit. Raw benchmark text, provider prompts and responses, credentials, embeddings,
checkpoints, and private data remain excluded.

## Build and Check

Run the full repository checks first, commit the intended release, and require a
clean worktree. Then create the export:

```powershell
uv run --extra dev python -m scripts.export_anonymous_artifact `
  --output tmp/anonymous_artifact
```

The exporter:

1. copies only `git ls-files` entries;
2. excludes Git history, CI metadata, caches, local configuration, raw/private data,
   provider material, and the compiled paper directory;
3. redacts private owner and repository identifiers in the copied text only;
4. repairs evidence-manifest hashes for redacted transformation scripts;
5. fails on common API-key and private-key shapes; and
6. writes `ANONYMIZATION_REPORT.json` with output hashes and no source commit ID.

Verify the exported package independently before publishing it:

```powershell
Push-Location tmp/anonymous_artifact
uv sync --extra dev --extra paper
uv run --extra dev python -m pytest -q
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python -m scripts.check_claim_contract
uv run --extra dev python -m scripts.verify_evidence
uv run --extra dev --extra paper python -m scripts.verify_paper
Pop-Location
```

Only after these checks pass should this directory be initialized as a new anonymous
Git repository. Do not add a remote, author identity, submission ID, or public URL to
the private source repository.
