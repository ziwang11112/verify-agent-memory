"""Publish or verify the content-free paired-exposure result bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "tmp" / "counterfactual_exposure" / "provider_runtime"
DEFAULT_OUTPUT = ROOT / "results" / "counterfactual_exposure"

EXECUTION_COMMIT = "82d3bce8023d1ccc97bb21b0bbb36e15a4b3c6af"
BASE_PROTOCOL_SHA256 = "b0190521ff8886201732702ec4324f4b5f68feb0a22f20d5d48e37ed84d913ff"
EXECUTION_PROTOCOL_SHA256 = "79d15dec9ac9ac315aa25b30877ea0c8a783f30bb667fb588acea244ddc74786"
UNLOCK_SHA256 = "a12b622cb12e8589732bd33a0057aad54648dd01aade35c3084f30bb5ce2ff99"
EXPECTED_REQUESTS = 384
EXPECTED_PAIRS = 192
EXPECTED_CELL_ROWS = 20
EXPECTED_BOOTSTRAP_ROWS = 25


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    slug: str


PROVIDERS = (
    Provider("OpenAI", "gpt-5.6-sol", "openai-gpt-5.6-sol"),
    Provider("Gemini", "gemini-3.6-flash", "gemini-gemini-3.6-flash"),
    Provider("DeepSeek", "deepseek-v4-pro", "deepseek-deepseek-v4-pro"),
)

FORBIDDEN_PUBLISHED_FIELDS = {
    "answer",
    "answer_text",
    "prompt",
    "query_text",
    "raw_response",
    "response",
    "response_text",
    "target_marker",
}


def _object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} contains a non-object row")
        rows.append(value)
    return rows


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"inconsistent fields for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _require_binding(value: Mapping[str, object], *, path: Path) -> None:
    recorded_base = value.get("base_protocol_sha256", value.get("protocol_sha256"))
    if recorded_base != BASE_PROTOCOL_SHA256:
        raise ValueError(f"base protocol drifted in {path}")
    recorded_execution = value.get("execution_protocol_sha256")
    if recorded_execution is not None and recorded_execution != EXECUTION_PROTOCOL_SHA256:
        raise ValueError(f"execution protocol drifted in {path}")


def _manifest_outputs(manifest: Mapping[str, object]) -> dict[str, str]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise TypeError("score manifest outputs must be an array")
    result: dict[str, str] = {}
    for raw in outputs:
        if not isinstance(raw, Mapping):
            raise TypeError("score manifest contains a malformed output")
        path = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise TypeError("score manifest output must have path and sha256")
        result[path] = digest
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = math.floor((len(ordered) - 1) * probability)
    return ordered[index]


def _validate_score_rows(
    provider: Provider,
    score_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pair_rows = _csv(score_dir / "pair_scores.csv")
    cell_rows = _csv(score_dir / "cell_metrics.csv")
    bootstrap_rows = _csv(score_dir / "bootstrap_ci.csv")
    if len(pair_rows) != EXPECTED_PAIRS:
        raise ValueError(f"{provider.name} has {len(pair_rows)} pairs")
    if len(cell_rows) != EXPECTED_CELL_ROWS:
        raise ValueError(f"{provider.name} has {len(cell_rows)} cell rows")
    if len(bootstrap_rows) != EXPECTED_BOOTSTRAP_ROWS:
        raise ValueError(f"{provider.name} has {len(bootstrap_rows)} bootstrap rows")
    if any(row.get("model") != provider.model for row in pair_rows):
        raise ValueError(f"{provider.name} pair-score model drifted")
    pair_output = [{"provider": provider.name, **row} for row in pair_rows]
    cell_output = [{"provider": provider.name, "model": provider.model, **row} for row in cell_rows]
    bootstrap_output = [
        {"provider": provider.name, "model": provider.model, **row} for row in bootstrap_rows
    ]
    return pair_output, cell_output, bootstrap_output


def _provider_bundle(
    runtime_dir: Path,
    provider: Provider,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    fixture_path = runtime_dir / "fixtures" / f"{provider.slug}.json"
    completion_path = runtime_dir / "completion" / f"{provider.slug}.json"
    raw_path = runtime_dir / "raw" / f"{provider.slug}.jsonl"
    score_dir = runtime_dir / "scores" / provider.slug
    score_manifest_path = score_dir / "manifest.json"
    execution_manifest_path = score_dir / "execution_manifest.json"

    fixture = _object(fixture_path)
    completion = _object(completion_path)
    score_manifest = _object(score_manifest_path)
    execution_manifest = _object(execution_manifest_path)
    for value, path in (
        (fixture, fixture_path),
        (completion, completion_path),
        (score_manifest, score_manifest_path),
        (execution_manifest, execution_manifest_path),
    ):
        _require_binding(value, path=path)
    if fixture.get("status") != "passed" or fixture.get("attempts") != 1:
        raise ValueError(f"{provider.name} fixture is not a one-attempt pass")
    if fixture.get("response_saved") is not False:
        raise ValueError(f"{provider.name} fixture unexpectedly saved response text")
    if completion.get("completed_requests") != EXPECTED_REQUESTS:
        raise ValueError(f"{provider.name} execution is incomplete")
    if completion.get("prefix_scoring") is not False:
        raise ValueError(f"{provider.name} completion permits prefix scoring")
    if score_manifest.get("status") != "complete_local_score_not_official_result":
        raise ValueError(f"{provider.name} score status drifted")
    if score_manifest.get("scored_pair_count") != EXPECTED_PAIRS:
        raise ValueError(f"{provider.name} score pair count drifted")
    if score_manifest.get("answer_text_published") is not False:
        raise ValueError(f"{provider.name} score manifest publishes answer text")
    if score_manifest.get("llm_judge") is not False:
        raise ValueError(f"{provider.name} score manifest records an LLM judge")
    if score_manifest.get("prefix_scoring") is not False:
        raise ValueError(f"{provider.name} score manifest permits prefix scoring")
    if execution_manifest.get("status") != "complete_local_score_not_official_result":
        raise ValueError(f"{provider.name} execution manifest status drifted")
    if execution_manifest.get("provider_call_made_during_scoring") is not False:
        raise ValueError(f"{provider.name} scoring made a provider call")
    for receipt in (fixture, completion, execution_manifest):
        if receipt.get("provider") != provider.name or receipt.get("model") != provider.model:
            raise ValueError(f"{provider.name} provider/model binding drifted")
    if score_manifest.get("model") != provider.model:
        raise ValueError(f"{provider.name} scorer-model binding drifted")

    raw_rows = _jsonl(raw_path)
    if len(raw_rows) != EXPECTED_REQUESTS:
        raise ValueError(f"{provider.name} raw checkpoint has {len(raw_rows)} rows")
    request_ids = [str(row.get("request_id")) for row in raw_rows]
    if len(set(request_ids)) != EXPECTED_REQUESTS:
        raise ValueError(f"{provider.name} request IDs are not unique")
    if any(row.get("attempts") != 1 for row in raw_rows):
        raise ValueError(f"{provider.name} checkpoint contains a retry")
    if sha256_file(raw_path) != completion.get("checkpoint_sha256"):
        raise ValueError(f"{provider.name} checkpoint hash drifted")
    if execution_manifest.get("checkpoint_sha256") != completion.get("checkpoint_sha256"):
        raise ValueError(f"{provider.name} scoring did not bind the completion checkpoint")

    score_outputs = _manifest_outputs(score_manifest)
    for name in ("pair_scores.csv", "cell_metrics.csv", "bootstrap_ci.csv"):
        path = score_dir / name
        if score_outputs.get(name) != sha256_file(path):
            raise ValueError(f"{provider.name} score output hash drifted: {name}")
    pairs, cells, bootstrap = _validate_score_rows(provider, score_dir)

    latencies = [float(row["latency_ms"]) for row in raw_rows]
    input_tokens = sum(int(row["usage"]["input_tokens"]) for row in raw_rows)  # type: ignore[index]
    output_tokens = sum(int(row["usage"]["output_tokens"]) for row in raw_rows)  # type: ignore[index]
    checkpoint_cost = sum(float(row["cap_accounted_cost_usd"]) for row in raw_rows)
    fixture_cost = float(fixture["cap_accounted_cost_usd"])
    total_cost = float(completion["cap_accounted_cost_usd"])
    if not math.isclose(checkpoint_cost + fixture_cost, total_cost, abs_tol=1e-9):
        raise ValueError(f"{provider.name} cost ledger does not reconcile")

    usage = {
        "provider": provider.name,
        "model": provider.model,
        "fixture_calls": 1,
        "scored_requests": EXPECTED_REQUESTS,
        "total_calls": EXPECTED_REQUESTS + 1,
        "max_attempts": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "mean_latency_ms": round(statistics.fmean(latencies), 6),
        "median_latency_ms": round(statistics.median(latencies), 6),
        "p95_latency_ms": round(_quantile(latencies, 0.95), 6),
        "fixture_cap_accounted_cost_usd": fixture_cost,
        "total_cap_accounted_cost_usd": total_cost,
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "response_bundle_sha256": score_manifest["response_bundle_sha256"],
        "score_manifest_sha256": execution_manifest["score_manifest_sha256"],
    }
    receipt = {
        "provider": provider.name,
        "model": provider.model,
        "fixture_sha256": sha256_file(fixture_path),
        "completion_sha256": sha256_file(completion_path),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "response_bundle_sha256": score_manifest["response_bundle_sha256"],
        "score_manifest_sha256": execution_manifest["score_manifest_sha256"],
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
    }
    return pairs, cells, bootstrap, usage, receipt


def _readme(
    cell_rows: Sequence[Mapping[str, object]], bootstrap_rows: Sequence[Mapping[str, object]]
) -> str:
    def cell(provider: str, name: str) -> Mapping[str, object]:
        return next(
            row
            for row in cell_rows
            if row["provider"] == provider and row["scope"] == "overall" and row["cell"] == name
        )

    def interval(provider: str, contrast: str) -> Mapping[str, object]:
        return next(
            row
            for row in bootstrap_rows
            if row["provider"] == provider
            and row["scope"] == "overall"
            and row["contrast"] == contrast
        )

    labels = {provider.name: provider.model for provider in PROVIDERS}
    lines = [
        "# Paired Counterfactual Exposure Intervention",
        "",
        "This is a controlled, hash-bound diagnostic, not an official benchmark result. One",
        "candidate is exposed or withheld while the query and all other candidates remain fixed.",
        "Provider estimates are reported separately and are never pooled.",
        "",
        (
            "| Reader | Relevant + admissible ATE | Relevant + inadmissible ATE | "
            "Selectivity gap (95% CI) |"
        ),
        "| --- | ---: | ---: | ---: |",
    ]
    gap_name = "relevant_admissible_minus_relevant_inadmissible_exposure_effect"
    for provider in PROVIDERS:
        admissible = float(cell(provider.name, "relevant_admissible")["exposure_effect"])
        inadmissible = float(cell(provider.name, "relevant_inadmissible")["exposure_effect"])
        gap = interval(provider.name, gap_name)
        lines.append(
            f"| `{labels[provider.name]}` | {admissible:.4f} | {inadmissible:.4f} | "
            f"{float(gap['estimate']):.4f} "
            f"[{float(gap['ci_lower']):.4f}, {float(gap['ci_upper']):.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Across all three frozen readers, exposure increased disclosure of relevant admissible",
            (
                "evidence substantially more than disclosure of relevant inadmissible evidence. "
                "DeepSeek"
            ),
            (
                "also showed a positive relevant-inadmissible exposure effect; the other two "
                "readers did"
            ),
            "not show an aggregate incremental effect in that cell. This supports a controlled",
            (
                "prompt-level exposure claim only. It does not estimate natural-corpus prevalence, "
                "latent"
            ),
            "metadata inference, production safety, or official benchmark performance.",
            "",
            (
                "The published bundle contains derived binary pair scores, aggregate metrics, "
                "uncertainty,"
            ),
            (
                "cost/latency summaries, and hashes. It contains no prompts, candidate text, "
                "literal target"
            ),
            "markers, provider responses, answer text, API credentials, or private data.",
        ]
    )
    return "\n".join(lines) + "\n"


def publish(runtime_dir: Path, output_dir: Path) -> Mapping[str, object]:
    """Validate the frozen execution and publish only content-free derivatives."""
    pairs: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    bootstrap: list[dict[str, object]] = []
    usage: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for provider in PROVIDERS:
        provider_pairs, provider_cells, provider_bootstrap, provider_usage, receipt = (
            _provider_bundle(runtime_dir, provider)
        )
        pairs.extend(provider_pairs)
        cells.extend(provider_cells)
        bootstrap.extend(provider_bootstrap)
        usage.append(provider_usage)
        receipts.append(receipt)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (
        output_dir / "pair_scores.csv",
        output_dir / "cell_metrics.csv",
        output_dir / "bootstrap_ci.csv",
        output_dir / "provider_usage.csv",
    )
    for path, rows in zip(output_paths, (pairs, cells, bootstrap, usage), strict=True):
        _write_csv(path, rows)
    readme_path = output_dir / "README.md"
    _write_text(readme_path, _readme(cells, bootstrap))
    published_paths = (*output_paths, readme_path)
    manifest = {
        "schema_version": 1,
        "status": "complete_controlled_diagnostic_not_official_benchmark_result",
        "execution_commit": EXECUTION_COMMIT,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "execution_protocol_sha256": EXECUTION_PROTOCOL_SHA256,
        "unlock_sha256": UNLOCK_SHA256,
        "comparison_eligible_providers": [
            {"provider": provider.name, "model": provider.model} for provider in PROVIDERS
        ],
        "request_count_per_provider": EXPECTED_REQUESTS,
        "fixture_calls_per_provider": 1,
        "scored_pairs_per_provider": EXPECTED_PAIRS,
        "total_provider_calls": sum(int(row["total_calls"]) for row in usage),
        "total_cap_accounted_cost_usd": sum(
            float(row["total_cap_accounted_cost_usd"]) for row in usage
        ),
        "provider_receipts": receipts,
        "outputs": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": sum(1 for _ in _csv(path)) if path.suffix == ".csv" else None,
            }
            for path in published_paths
        ],
        "publisher": "scripts/publish_counterfactual_exposure_results.py",
        "publisher_sha256": sha256_file(Path(__file__)),
        "raw_provider_responses_published": False,
        "answer_text_published": False,
        "prompt_or_candidate_text_published": False,
        "llm_judge": False,
        "transport_retries": 0,
        "output_repair": False,
        "selective_rerun": False,
        "model_pooling": False,
        "official_result": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    validate_published(output_dir)
    return manifest


def validate_published(output_dir: Path) -> Mapping[str, object]:
    """Validate receipts, row counts, and the content-free publication boundary."""
    manifest = _object(output_dir / "manifest.json")
    if manifest.get("status") != "complete_controlled_diagnostic_not_official_benchmark_result":
        raise ValueError("published result status drifted")
    if manifest.get("official_result") is not False:
        raise ValueError("published result is incorrectly marked official")
    for field in (
        "raw_provider_responses_published",
        "answer_text_published",
        "prompt_or_candidate_text_published",
        "llm_judge",
        "output_repair",
        "selective_rerun",
        "model_pooling",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"publication boundary drifted: {field}")
    expected_rows = {
        "pair_scores.csv": EXPECTED_PAIRS * len(PROVIDERS),
        "cell_metrics.csv": EXPECTED_CELL_ROWS * len(PROVIDERS),
        "bootstrap_ci.csv": EXPECTED_BOOTSTRAP_ROWS * len(PROVIDERS),
        "provider_usage.csv": len(PROVIDERS),
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise TypeError("published manifest outputs must be an array")
    for raw in outputs:
        if not isinstance(raw, Mapping):
            raise TypeError("published manifest contains a malformed output")
        name = str(raw["path"])
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != raw.get("sha256"):
            raise ValueError(f"published output failed its receipt: {name}")
        if path.suffix == ".csv":
            rows = _csv(path)
            if len(rows) != expected_rows[name]:
                raise ValueError(f"published row count drifted: {name}")
            fields = set(rows[0]) if rows else set()
            if fields.intersection(FORBIDDEN_PUBLISHED_FIELDS):
                raise ValueError(f"published output contains forbidden fields: {name}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "verify"))
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "publish":
        manifest = publish(args.runtime_dir.resolve(), args.output_dir.resolve())
    else:
        manifest = validate_published(args.output_dir.resolve())
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
