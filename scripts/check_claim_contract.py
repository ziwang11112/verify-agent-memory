"""Validate the machine-readable retrieval-admissibility claim contract."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REQUIRED_CLAIM_FIELDS = {
    "id",
    "status",
    "claim",
    "estimand",
    "population",
    "source_repository",
    "source_snapshot_commit",
    "execution_commit",
    "source_artifacts",
    "exact_values",
    "confidence_intervals",
    "allowed_wording",
    "forbidden_wording",
    "known_limitations",
    "paper_locations",
    "verification_tests",
}

REQUIRED_INTERVAL_FIELDS = {"metric", "estimate", "lower", "upper"}
GATEMEM_CLAIM_IDS = {"C2", "C3"}
REQUIRED_SOURCE_FIELDS = {
    "id",
    "category",
    "old_repository",
    "frozen_commit",
    "source_path",
    "source_sha256",
    "intended_normalized_destination",
    "transformation_required",
    "public_paper_label",
    "contains_raw_text_or_private_content",
}
SOURCE_CATEGORIES = {
    "PORT_AND_REFACTOR",
    "IMPORT_AS_FROZEN_AGGREGATE",
    "GENERATED_FROM_HASH_BOUND_EXECUTION",
    "GENERATED_FROM_FROZEN_POSTHOC",
    "GENERATED_FROM_PUBLIC_DIAGNOSTIC",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MIGRATION_SNAPSHOT = "a28093110325968c26906223e9eb0f1e078f6aad"
NATURAL_EXECUTION_COMMIT = "8e34e3d41c56e1699696bc27be95cdac7c9528e5"
COUNTERFACTUAL_EXPOSURE_EXECUTION_COMMIT = "82d3bce8023d1ccc97bb21b0bbb36e15a4b3c6af"
CLAUDE_OPUS5_EXPOSURE_EXECUTION_COMMIT = "938320909b4c2d13e286987dc4297a7cb6ef73a7"
NATURAL_POSTHOC_IMPLEMENTATION_COMMIT = "009ca3657fb9ebe2bad2f107d5f374c69a4afab3"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _walk_numbers(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            errors.append(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_numbers(child, f"{path}.{key}", errors)
        return
    if _is_sequence(value):
        for index, child in enumerate(value):
            _walk_numbers(child, f"{path}[{index}]", errors)


def _require_nonempty_list(
    claim: Mapping[str, Any], field: str, claim_id: str, errors: list[str]
) -> None:
    value = claim.get(field)
    if not _is_sequence(value) or not value:
        errors.append(f"{claim_id}.{field} must be a non-empty list")


def _combined_text(value: Any) -> str:
    if _is_sequence(value):
        return " ".join(str(item) for item in value).lower()
    return str(value).lower()


def _validate_intervals(claim: Mapping[str, Any], claim_id: str, errors: list[str]) -> None:
    intervals = claim.get("confidence_intervals")
    if not _is_sequence(intervals):
        errors.append(f"{claim_id}.confidence_intervals must be a list")
        return

    for index, interval in enumerate(intervals):
        path = f"{claim_id}.confidence_intervals[{index}]"
        if not isinstance(interval, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        missing = REQUIRED_INTERVAL_FIELDS - interval.keys()
        if missing:
            errors.append(f"{path} missing fields: {', '.join(sorted(missing))}")
            continue
        estimate = interval["estimate"]
        lower = interval["lower"]
        upper = interval["upper"]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (estimate, lower, upper)
        ):
            errors.append(f"{path} estimate, lower, and upper must be numeric")
            continue
        if not all(math.isfinite(value) for value in (estimate, lower, upper)):
            errors.append(f"{path} estimate, lower, and upper must be finite")
            continue
        if lower > upper:
            errors.append(f"{path} has lower > upper")
        elif not lower <= estimate <= upper:
            errors.append(f"{path} does not contain its estimate")


def validate_contract(data: Any, readable_contract: str | None = None) -> list[str]:
    """Return all validation errors for a parsed claim contract."""
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ["contract root must be a mapping"]

    claims = data.get("claims")
    if not _is_sequence(claims) or not claims:
        return ["claims must be a non-empty list"]

    claim_by_id: dict[str, Mapping[str, Any]] = {}
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            errors.append(f"claims[{index}] must be a mapping")
            continue

        claim_id = str(claim.get("id", f"claims[{index}]"))
        missing = REQUIRED_CLAIM_FIELDS - claim.keys()
        if missing:
            errors.append(f"{claim_id} missing fields: {', '.join(sorted(missing))}")

        if claim_id in claim_by_id:
            errors.append(f"duplicate claim id: {claim_id}")
        else:
            claim_by_id[claim_id] = claim

        _walk_numbers(claim.get("exact_values"), f"{claim_id}.exact_values", errors)
        _validate_intervals(claim, claim_id, errors)

        status = claim.get("status")
        if status != "definition":
            _require_nonempty_list(claim, "source_artifacts", claim_id, errors)
            _require_nonempty_list(claim, "allowed_wording", claim_id, errors)
            _require_nonempty_list(claim, "forbidden_wording", claim_id, errors)
            if not claim.get("source_repository"):
                errors.append(f"{claim_id}.source_repository is required for empirical claims")
            if not claim.get("source_snapshot_commit"):
                errors.append(f"{claim_id}.source_snapshot_commit is required for empirical claims")

    smoke = claim_by_id.get("C4")
    full = claim_by_id.get("C5")
    if smoke and full and smoke.get("population") == full.get("population"):
        errors.append("C4 and C5 must not use identical populations")

    if full:
        exact_values = full.get("exact_values")
        wording = _combined_text(full.get("allowed_wording"))
        if not isinstance(exact_values, Mapping) or (
            "penalized_non_usable_upper_risk_delta" not in exact_values
        ):
            errors.append("C5 must name the frozen v1 penalized non-usable upper risk")
        if "non-usable" not in wording:
            errors.append("C5 allowed wording must identify the non-usable risk family")

    historical = claim_by_id.get("C7")
    if historical:
        limitations = _combined_text(historical.get("known_limitations"))
        if historical.get("status") != "frozen_historical_released_field_v1":
            errors.append("C7 must remain the frozen historical released-field v1 claim")
        if "v1 arm" not in limitations or "corrected v2" not in limitations:
            errors.append("C7 must distinguish the historical v1 and corrected v2 semantics")

    for claim_id in GATEMEM_CLAIM_IDS:
        claim = claim_by_id.get(claim_id)
        if not claim:
            continue
        limitations = _combined_text(claim.get("known_limitations"))
        if "non-causal" not in limitations:
            errors.append(f"{claim_id} must include a non-causal limitation")
        if "same-provider" not in limitations:
            errors.append(f"{claim_id} must include a same-provider limitation")

    exposure = claim_by_id.get("C8")
    if exposure:
        limitations = _combined_text(exposure.get("known_limitations"))
        if exposure.get("status") != "controlled_prompt_intervention":
            errors.append("C8 must remain a controlled prompt intervention")
        exact_values = exposure.get("exact_values")
        if not isinstance(exact_values, Mapping) or exact_values.get("model_pooling") is not False:
            errors.append("C8.model_pooling must be false")
        if "constructed" not in limitations:
            errors.append("C8 must include a constructed-scenario limitation")
        if "never pooled" not in limitations:
            errors.append("C8 must include a no-pooling limitation")

    fixed_budget = claim_by_id.get("C9")
    if fixed_budget:
        limitations = _combined_text(fixed_budget.get("known_limitations"))
        if fixed_budget.get("status") != "post_hoc_frozen_ranking_decomposition":
            errors.append("C9 must remain a post-hoc frozen-ranking decomposition")
        if "post-hoc" not in limitations or "no settings were retuned" not in limitations:
            errors.append("C9 must include post-hoc and no-retuning boundaries")
        if "source-clustered" not in limitations:
            errors.append("C9 must identify the bootstrap unit")

    metadata = claim_by_id.get("C10")
    if metadata:
        limitations = _combined_text(metadata.get("known_limitations"))
        wording = _combined_text(metadata.get("allowed_wording"))
        if metadata.get("status") != "post_hoc_metadata_robustness_diagnostic":
            errors.append("C10 must remain a post-hoc metadata-robustness diagnostic")
        if "observed brackets" not in limitations or "not population thresholds" not in limitations:
            errors.append("C10 must preserve the observed-grid boundary")
        if "aggregate joint dominance does not imply zero" not in limitations:
            errors.append("C10 must distinguish aggregate dominance from zero leakage")
        if "corrected v2 attribution" not in wording or "policy metadata" not in wording:
            errors.append("C10 must preserve corrected-v2 policy attribution")

    inference = claim_by_id.get("C11")
    if inference:
        limitations = _combined_text(inference.get("known_limitations"))
        if inference.get("status") != "public_development_inference_gap_diagnostic":
            errors.append("C11 must remain a public-development inference-gap diagnostic")
        if "populations are separate" not in limitations or "never pooled" not in limitations:
            errors.append("C11 must keep natural and controlled populations separate")
        if "model estimates are reported separately" not in limitations:
            errors.append("C11 must keep model estimates separate")

    opus_replication = claim_by_id.get("C12")
    if opus_replication:
        limitations = _combined_text(opus_replication.get("known_limitations"))
        if opus_replication.get("status") != "controlled_prompt_reader_replication":
            errors.append("C12 must remain a controlled prompt reader replication")
        exact_values = opus_replication.get("exact_values")
        if not isinstance(exact_values, Mapping) or exact_values.get("model_pooling") is not False:
            errors.append("C12.model_pooling must be false")
        if "separate reader replication" not in limitations:
            errors.append("C12 must preserve its separate-execution boundary")
        if "constructed" not in limitations:
            errors.append("C12 must include a constructed-scenario limitation")
        if "never pooled" not in limitations:
            errors.append("C12 must include a no-pooling limitation")

    if readable_contract is not None:
        for claim_id in claim_by_id:
            if f"### {claim_id}:" not in readable_contract:
                errors.append(f"CLAIM_CONTRACT.md does not render {claim_id}")

    return errors


def validate_source_index(source_data: Any, contract_data: Any) -> list[str]:
    """Validate frozen source identities and coverage of empirical claim artifacts."""
    errors: list[str] = []
    if not isinstance(source_data, Mapping):
        return ["source index root must be a mapping"]

    artifacts = source_data.get("artifacts")
    if not _is_sequence(artifacts) or not artifacts:
        return ["source index artifacts must be a non-empty list"]

    ids: set[str] = set()
    paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            errors.append(f"source artifacts[{index}] must be a mapping")
            continue
        artifact_id = str(artifact.get("id", f"source artifacts[{index}]"))
        missing = REQUIRED_SOURCE_FIELDS - artifact.keys()
        if missing:
            errors.append(f"{artifact_id} missing source fields: {', '.join(sorted(missing))}")

        if artifact_id in ids:
            errors.append(f"duplicate source artifact id: {artifact_id}")
        ids.add(artifact_id)

        source_path = artifact.get("source_path")
        if source_path in paths:
            errors.append(f"duplicate source artifact path: {source_path}")
        if isinstance(source_path, str):
            paths.add(source_path)
        else:
            errors.append(f"{artifact_id}.source_path must be a string")

        source_hash = artifact.get("source_sha256")
        if not isinstance(source_hash, str) or not SHA256_PATTERN.fullmatch(source_hash):
            errors.append(f"{artifact_id}.source_sha256 must be a lowercase SHA-256")
        category = artifact.get("category")
        if category == "GENERATED_FROM_HASH_BOUND_EXECUTION":
            if artifact.get("frozen_commit") not in {
                COUNTERFACTUAL_EXPOSURE_EXECUTION_COMMIT,
                CLAUDE_OPUS5_EXPOSURE_EXECUTION_COMMIT,
            }:
                errors.append(f"{artifact_id}.frozen_commit must match a paired-exposure execution")
        elif category == "GENERATED_FROM_FROZEN_POSTHOC":
            if artifact.get("frozen_commit") != NATURAL_POSTHOC_IMPLEMENTATION_COMMIT:
                errors.append(
                    f"{artifact_id}.frozen_commit must match the natural-posthoc implementation"
                )
        elif category == "GENERATED_FROM_PUBLIC_DIAGNOSTIC":
            frozen_commit = artifact.get("frozen_commit")
            if not isinstance(frozen_commit, str) or not COMMIT_PATTERN.fullmatch(frozen_commit):
                errors.append(f"{artifact_id}.frozen_commit must be a lowercase Git commit")
        elif artifact.get("frozen_commit") != MIGRATION_SNAPSHOT:
            errors.append(f"{artifact_id}.frozen_commit must match the migration snapshot")
        if category not in SOURCE_CATEGORIES:
            errors.append(f"{artifact_id}.category is not allowed")
        if not isinstance(artifact.get("contains_raw_text_or_private_content"), bool):
            errors.append(f"{artifact_id}.contains_raw_text_or_private_content must be boolean")

    if isinstance(contract_data, Mapping) and _is_sequence(contract_data.get("claims")):
        for claim in contract_data["claims"]:
            if not isinstance(claim, Mapping) or claim.get("status") == "definition":
                continue
            claim_id = str(claim.get("id", "unknown"))
            source_artifacts = claim.get("source_artifacts", [])
            if not _is_sequence(source_artifacts):
                continue
            for source_path in source_artifacts:
                if source_path not in paths:
                    errors.append(f"{claim_id} source artifact is not indexed: {source_path}")

    return errors


def validate_governance_documents(provenance: str, allowlist: str) -> list[str]:
    """Check that the human-readable migration boundaries remain explicit."""
    errors: list[str] = []
    for identity in (MIGRATION_SNAPSHOT, NATURAL_EXECUTION_COMMIT):
        if identity not in provenance:
            errors.append(f"PROVENANCE.md is missing identity {identity}")

    required_sections = {
        "## PORT_AND_REFACTOR",
        "## IMPORT_AS_FROZEN_AGGREGATE",
        "## REFERENCE_ONLY",
        "## PROHIBITED",
    }
    for section in sorted(required_sections):
        if section not in allowlist:
            errors.append(f"MIGRATION_ALLOWLIST.md is missing {section}")
    return errors


def load_contract(path: Path) -> Any:
    """Load YAML without accepting an empty document."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims",
        type=Path,
        default=root / "claims" / "claims.yaml",
        help="Path to claims YAML.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=root / "CLAIM_CONTRACT.md",
        help="Path to readable claim contract.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=root / "SOURCE_ARTIFACTS.yaml",
        help="Path to source artifact index.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=root / "MIGRATION_ALLOWLIST.md",
        help="Path to migration allowlist.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=root / "PROVENANCE.md",
        help="Path to provenance record.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = load_contract(args.claims)
        source_data = load_contract(args.sources)
        readable_contract = args.contract.read_text(encoding="utf-8")
        allowlist = args.allowlist.read_text(encoding="utf-8")
        provenance = args.provenance.read_text(encoding="utf-8")
    except (OSError, yaml.YAMLError) as exc:
        print(f"claim contract could not be loaded: {exc}", file=sys.stderr)
        return 1

    errors = validate_contract(data, readable_contract)
    errors.extend(validate_source_index(source_data, data))
    errors.extend(validate_governance_documents(provenance, allowlist))
    if errors:
        print("claim contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "claim contract valid: "
        f"{len(data['claims'])} claims, {len(source_data['artifacts'])} source artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
