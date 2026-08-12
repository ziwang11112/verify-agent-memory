"""Normalize the published paired-exposure aggregates into claim-bound evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.publish_counterfactual_exposure_results import (
    EXECUTION_COMMIT,
    validate_published,
)
from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "counterfactual_exposure"
DEFAULT_NORMALIZED = ROOT / "evidence" / "normalized" / "counterfactual_exposure.csv"
DEFAULT_MANIFEST = ROOT / "evidence" / "manifests" / "counterfactual_exposure.json"
SOURCE_REPOSITORY = "ziwang11112/verify-agent-memory"
GAP_CONTRAST = "relevant_admissible_minus_relevant_inadmissible_exposure_effect"
CELL_CONTRASTS = (
    "relevant_admissible",
    "relevant_inadmissible",
    "irrelevant_admissible",
    "irrelevant_inadmissible",
)
PROVIDERS = (
    ("OpenAI", "gpt-5.6-sol"),
    ("Gemini", "gemini-3.6-flash"),
    ("DeepSeek", "deepseek-v4-pro"),
)
FIELDS = (
    "claim_id",
    "family",
    "population",
    "source",
    "contrast",
    "metric",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "n",
    "notes",
)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Sequence[Mapping[str, str]], **selectors: str) -> Mapping[str, str]:
    matches = [
        row for row in rows if all(row.get(field) == value for field, value in selectors.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {selectors!r}, found {len(matches)}")
    return matches[0]


def normalized_rows(results_dir: Path) -> tuple[dict[str, object], ...]:
    """Return all preregistered overall effects without pooling readers."""
    validate_published(results_dir)
    intervals = _csv(results_dir / "bootstrap_ci.csv")
    rows = []
    population = "16 controlled scenarios with paired exposed and withheld reader requests"
    notes = (
        "controlled_prompt_intervention;scenario_bootstrap;providers_not_pooled;"
        "no_judge;not_official_benchmark"
    )
    for provider, model in PROVIDERS:
        for contrast in (*CELL_CONTRASTS, GAP_CONTRAST):
            source = _one(
                intervals,
                provider=provider,
                model=model,
                scope="overall",
                contrast=contrast,
            )
            metric = "selectivity_gap" if contrast == GAP_CONTRAST else f"{contrast}_effect"
            rows.append(
                {
                    "claim_id": "C8",
                    "family": "counterfactual_exposure",
                    "population": population,
                    "source": model,
                    "contrast": "admissible_minus_inadmissible"
                    if contrast == GAP_CONTRAST
                    else "exposed_minus_withheld",
                    "metric": metric,
                    "estimate": source["estimate"],
                    "ci95_lower": source["ci_lower"],
                    "ci95_upper": source["ci_upper"],
                    "n": source["unit_count"],
                    "notes": notes,
                }
            )
    return tuple(rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(
    results_dir: Path,
    normalized_path: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    rows = normalized_rows(results_dir)
    if len(rows) != 15:
        raise ValueError(f"expected 15 normalized rows, found {len(rows)}")
    _write_csv(normalized_path, rows)
    source_names = (
        "manifest.json",
        "cell_metrics.csv",
        "bootstrap_ci.csv",
        "provider_usage.csv",
    )
    manifest = {
        "schema_version": 1,
        "contains_raw_text_or_private_content": False,
        "normalized_file": normalized_path.relative_to(ROOT).as_posix(),
        "normalized_sha256": sha256_file(normalized_path),
        "row_count": len(rows),
        "source_artifacts": [
            {
                "path": (results_dir / name).relative_to(ROOT).as_posix(),
                "sha256": sha256_file(results_dir / name),
            }
            for name in source_names
        ],
        "source_repository": SOURCE_REPOSITORY,
        "source_snapshot_commit": EXECUTION_COMMIT,
        "transformation_script": Path(__file__).relative_to(ROOT).as_posix(),
        "transformation_script_sha256": sha256_file(Path(__file__)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(
        args.results_dir.resolve(),
        args.normalized.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
