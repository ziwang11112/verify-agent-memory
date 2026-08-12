# Security and Data Handling

Do not report a secret by opening a public issue. Contact the repository maintainer
privately through the hosting platform's security-reporting channel.

## Never Commit

- API keys, tokens, private keys, or `.env` files;
- private user or organization data;
- raw model/provider responses containing benchmark payloads;
- hidden evaluation labels or unreleased benchmark splits;
- embedding shards, route checkpoints, or private scored derivatives; or
- third-party data without confirmed redistribution rights.

The CI reproducibility audit scans tracked files for common credential shapes and
forbids known private/raw paths. This is a guardrail, not a replacement for manual
review.

## Supported Scope

Security reports should concern this repository's evaluation library, execution
guards, data-boundary checks, or release tooling. Vulnerabilities in upstream
benchmarks or model providers should be reported to their respective maintainers.
