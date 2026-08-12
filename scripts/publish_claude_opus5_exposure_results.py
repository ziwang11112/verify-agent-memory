"""Publish or verify the content-free Claude Opus 5 exposure replication."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from scripts import publish_counterfactual_exposure_results as base
from verify_agent_memory.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "tmp" / "counterfactual_exposure" / "claude_opus5_replication"
DEFAULT_OUTPUT = ROOT / "results" / "claude_opus5_exposure_replication"

EXECUTION_COMMIT = "938320909b4c2d13e286987dc4297a7cb6ef73a7"
BASE_PROTOCOL_SHA256 = "b0190521ff8886201732702ec4324f4b5f68feb0a22f20d5d48e37ed84d913ff"
EXECUTION_PROTOCOL_SHA256 = "a01c17e1bff6dc00f7ba631c24dfc0038334805f06a9d4a0ee1583bff7efc11e"
UNLOCK_SHA256 = "37d048c60e64a53a43b8b38729fc491b539e397c19d4f74c98d736272d803d37"
PROVIDER = base.Provider("Anthropic", "claude-opus-5", "anthropic-claude-opus-5")


@contextmanager
def _replication_binding() -> Iterator[None]:
    previous = (
        base.BASE_PROTOCOL_SHA256,
        base.EXECUTION_PROTOCOL_SHA256,
        base.UNLOCK_SHA256,
    )
    base.BASE_PROTOCOL_SHA256 = BASE_PROTOCOL_SHA256
    base.EXECUTION_PROTOCOL_SHA256 = EXECUTION_PROTOCOL_SHA256
    base.UNLOCK_SHA256 = UNLOCK_SHA256
    try:
        yield
    finally:
        (
            base.BASE_PROTOCOL_SHA256,
            base.EXECUTION_PROTOCOL_SHA256,
            base.UNLOCK_SHA256,
        ) = previous


def _readme(
    cells: Sequence[Mapping[str, object]],
    intervals: Sequence[Mapping[str, object]],
) -> str:
    def cell(name: str) -> Mapping[str, object]:
        return next(row for row in cells if row["scope"] == "overall" and row["cell"] == name)

    def interval(name: str) -> Mapping[str, object]:
        return next(
            row for row in intervals if row["scope"] == "overall" and row["contrast"] == name
        )

    admissible = cell("relevant_admissible")
    inadmissible = cell("relevant_inadmissible")
    gap = interval("relevant_admissible_minus_relevant_inadmissible_exposure_effect")
    inadmissible_ci = interval("relevant_inadmissible")
    reader_row = (
        f"| `claude-opus-5` | {float(admissible['exposure_effect']):.4f} | "
        f"{float(inadmissible['exposure_effect']):.4f} | {float(gap['estimate']):.4f} "
        f"[{float(gap['ci_lower']):.4f}, {float(gap['ci_upper']):.4f}] |"
    )
    return f"""# Claude Opus 5 Paired-Exposure Replication

This is a separately executed, controlled reader replication. It reuses the frozen
16 scenarios, 192 paired units, and exposed/withheld construction from the original
three-reader experiment. It does not alter or pool the GPT-5.6 Sol, Gemini 3.6 Flash,
or DeepSeek V4 Pro estimates, and it is not an official benchmark result.

| Reader | Relevant + admissible ATE | Relevant + inadmissible ATE | Selectivity gap (95% CI) |
| --- | ---: | ---: | ---: |
{reader_row}

The relevant-inadmissible exposure effect is
{float(inadmissible_ci["estimate"]):.4f}
[{float(inadmissible_ci["ci_lower"]):.4f}, {float(inadmissible_ci["ci_upper"]):.4f}].
The selectivity-gap interval excludes zero, replicating the qualitative prompt-level
pattern with a fourth provider. The nonzero inadmissible point estimate also preserves
the paper's central boundary: reader restraint is imperfect after inadmissible content
has entered the prompt.

The execution used exactly `claude-opus-5`, thinking disabled, medium effort, one
accepted response per request, no transport retry, no output repair, no selective
rerun, and no judge. The content-free publication contains derived pair scores,
aggregate metrics, uncertainty, cost/latency summaries, and provenance hashes. It
contains no prompts, candidate text, literal target markers, provider responses,
answer text, credentials, or private data.
"""


def publish(runtime_dir: Path, output_dir: Path) -> Mapping[str, object]:
    """Validate the private execution and publish only content-free derivatives."""
    with _replication_binding():
        pairs, cells, bootstrap, usage, receipt = base._provider_bundle(runtime_dir, PROVIDER)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = (
        (output_dir / "pair_scores.csv", pairs),
        (output_dir / "cell_metrics.csv", cells),
        (output_dir / "bootstrap_ci.csv", bootstrap),
        (output_dir / "provider_usage.csv", [usage]),
    )
    for path, rows in tables:
        base._write_csv(path, rows)
    readme_path = output_dir / "README.md"
    base._write_text(readme_path, _readme(cells, bootstrap))
    published_paths = tuple(path for path, _rows in tables) + (readme_path,)
    manifest = {
        "schema_version": 1,
        "status": "complete_controlled_reader_replication_not_official_benchmark_result",
        "execution_commit": EXECUTION_COMMIT,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "execution_protocol_sha256": EXECUTION_PROTOCOL_SHA256,
        "unlock_sha256": UNLOCK_SHA256,
        "comparison_eligible_provider": {
            "provider": PROVIDER.name,
            "model": PROVIDER.model,
        },
        "relation_to_frozen_panel": "separate_fourth_reader_replication_no_pooling",
        "request_count": base.EXPECTED_REQUESTS,
        "fixture_calls": 1,
        "scored_pairs": base.EXPECTED_PAIRS,
        "total_provider_calls": int(usage["total_calls"]),
        "total_cap_accounted_cost_usd": float(usage["total_cap_accounted_cost_usd"]),
        "provider_receipt": receipt,
        "outputs": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(base._csv(path)) if path.suffix == ".csv" else None,
            }
            for path in published_paths
        ],
        "publisher": "scripts/publish_claude_opus5_exposure_results.py",
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
    base._write_json(output_dir / "manifest.json", manifest)
    validate_published(output_dir)
    return manifest


def validate_published(output_dir: Path) -> Mapping[str, object]:
    """Validate row counts, hashes, and publication boundaries."""
    manifest = base._object(output_dir / "manifest.json")
    if (
        manifest.get("status")
        != "complete_controlled_reader_replication_not_official_benchmark_result"
    ):
        raise ValueError("replication status drifted")
    expected_identity = {
        "provider": PROVIDER.name,
        "model": PROVIDER.model,
    }
    if manifest.get("comparison_eligible_provider") != expected_identity:
        raise ValueError("replication provider identity drifted")
    if manifest.get("execution_commit") != EXECUTION_COMMIT:
        raise ValueError("replication execution commit drifted")
    if manifest.get("execution_protocol_sha256") != EXECUTION_PROTOCOL_SHA256:
        raise ValueError("replication execution protocol drifted")
    if manifest.get("unlock_sha256") != UNLOCK_SHA256:
        raise ValueError("replication unlock drifted")
    if manifest.get("relation_to_frozen_panel") != "separate_fourth_reader_replication_no_pooling":
        raise ValueError("replication relation to frozen panel drifted")
    for field in (
        "raw_provider_responses_published",
        "answer_text_published",
        "prompt_or_candidate_text_published",
        "llm_judge",
        "output_repair",
        "selective_rerun",
        "model_pooling",
        "official_result",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"replication publication boundary drifted: {field}")
    expected_rows = {
        "pair_scores.csv": base.EXPECTED_PAIRS,
        "cell_metrics.csv": base.EXPECTED_CELL_ROWS,
        "bootstrap_ci.csv": base.EXPECTED_BOOTSTRAP_ROWS,
        "provider_usage.csv": 1,
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise TypeError("replication outputs must be an array")
    for raw in outputs:
        if not isinstance(raw, Mapping):
            raise TypeError("replication output receipt is malformed")
        name = str(raw["path"])
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != raw.get("sha256"):
            raise ValueError(f"replication output failed its receipt: {name}")
        if path.suffix == ".csv":
            rows = base._csv(path)
            if len(rows) != expected_rows[name]:
                raise ValueError(f"replication row count drifted: {name}")
            if rows and set(rows[0]).intersection(base.FORBIDDEN_PUBLISHED_FIELDS):
                raise ValueError(f"replication output contains forbidden fields: {name}")
    if manifest.get("publisher_sha256") != sha256_file(Path(__file__)):
        raise ValueError("replication publisher hash drifted")
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
