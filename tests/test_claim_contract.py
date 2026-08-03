from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from scripts.check_claim_contract import (
    validate_contract,
    validate_governance_documents,
    validate_source_index,
)

ROOT = Path(__file__).resolve().parents[1]


def load_valid_contract() -> dict:
    with (ROOT / "claims" / "claims.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_valid_source_index() -> dict:
    with (ROOT / "SOURCE_ARTIFACTS.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def claim(data: dict, claim_id: str) -> dict:
    return next(item for item in data["claims"] if item["id"] == claim_id)


def validate(data: dict) -> list[str]:
    readable = (ROOT / "CLAIM_CONTRACT.md").read_text(encoding="utf-8")
    return validate_contract(data, readable)


def test_repository_contract_passes() -> None:
    assert validate(load_valid_contract()) == []


def test_definition_has_no_empirical_source_requirement() -> None:
    data = load_valid_contract()
    definition = claim(data, "C1")
    assert definition["source_artifacts"] == []
    assert validate(data) == []


def test_missing_required_field_is_rejected() -> None:
    data = load_valid_contract()
    del claim(data, "C5")["estimand"]
    assert any("C5 missing fields: estimand" in error for error in validate(data))


def test_duplicate_ids_are_rejected() -> None:
    data = load_valid_contract()
    duplicate = deepcopy(claim(data, "C7"))
    duplicate["claim"] = "Duplicate fixture."
    data["claims"].append(duplicate)
    assert "duplicate claim id: C7" in validate(data)


def test_nonfinite_exact_value_is_rejected() -> None:
    data = load_valid_contract()
    claim(data, "C5")["exact_values"]["recall_delta"] = float("nan")
    assert any("C5.exact_values.recall_delta must be finite" in error for error in validate(data))


def test_nonfinite_interval_is_rejected() -> None:
    data = load_valid_contract()
    claim(data, "C5")["confidence_intervals"][0]["upper"] = float("inf")
    assert any("must be finite" in error for error in validate(data))


def test_reversed_interval_is_rejected() -> None:
    data = load_valid_contract()
    interval = claim(data, "C5")["confidence_intervals"][0]
    interval["lower"], interval["upper"] = interval["upper"], interval["lower"]
    assert any("has lower > upper" in error for error in validate(data))


def test_interval_must_contain_estimate() -> None:
    data = load_valid_contract()
    claim(data, "C5")["confidence_intervals"][0]["estimate"] = 5.0
    assert any("does not contain its estimate" in error for error in validate(data))


def test_empirical_claim_requires_source_artifacts() -> None:
    data = load_valid_contract()
    claim(data, "C6")["source_artifacts"] = []
    assert "C6.source_artifacts must be a non-empty list" in validate(data)


def test_empirical_claim_requires_wording_boundaries() -> None:
    data = load_valid_contract()
    claim(data, "C6")["forbidden_wording"] = []
    assert "C6.forbidden_wording must be a non-empty list" in validate(data)


def test_smoke_and_full_populations_are_distinct() -> None:
    data = load_valid_contract()
    claim(data, "C4")["population"] = claim(data, "C5")["population"]
    assert "C4 and C5 must not use identical populations" in validate(data)


def test_full_natural_claim_names_non_usable_risk_family() -> None:
    data = load_valid_contract()
    full = claim(data, "C5")
    full["exact_values"]["penalized_contamination_upper_delta"] = full["exact_values"].pop(
        "penalized_non_usable_upper_risk_delta"
    )
    assert "C5 must name the frozen v1 penalized non-usable upper risk" in validate(data)


def test_historical_arm_claim_distinguishes_v1_and_v2() -> None:
    data = load_valid_contract()
    claim(data, "C7")["known_limitations"] = ["Released-field upper bound."]
    assert "C7 must distinguish the historical v1 and corrected v2 semantics" in validate(data)


def test_gatemem_claim_requires_non_causal_limitation() -> None:
    data = load_valid_contract()
    gate_claim = claim(data, "C2")
    gate_claim["known_limitations"] = [
        item for item in gate_claim["known_limitations"] if "non-causal" not in item
    ]
    assert "C2 must include a non-causal limitation" in validate(data)


def test_gatemem_claim_requires_same_provider_limitation() -> None:
    data = load_valid_contract()
    gate_claim = claim(data, "C3")
    gate_claim["known_limitations"] = [
        item for item in gate_claim["known_limitations"] if "same-provider" not in item
    ]
    assert "C3 must include a same-provider limitation" in validate(data)


def test_counterfactual_exposure_claim_requires_controlled_boundary() -> None:
    data = load_valid_contract()
    exposure = claim(data, "C8")
    exposure["known_limitations"] = [
        item for item in exposure["known_limitations"] if "constructed" not in item
    ]
    assert "C8 must include a constructed-scenario limitation" in validate(data)


def test_counterfactual_exposure_claim_requires_no_pooling_boundary() -> None:
    data = load_valid_contract()
    exposure = claim(data, "C8")
    exposure["known_limitations"] = [
        item for item in exposure["known_limitations"] if "never pooled" not in item
    ]
    assert "C8 must include a no-pooling limitation" in validate(data)


def test_counterfactual_exposure_claim_forbids_model_pooling() -> None:
    data = load_valid_contract()
    claim(data, "C8")["exact_values"]["model_pooling"] = None
    assert "C8.model_pooling must be false" in validate(data)


def test_readable_contract_must_render_every_claim() -> None:
    data = load_valid_contract()
    assert "CLAIM_CONTRACT.md does not render C7" in validate_contract(data, "### C1: only")


def test_source_index_covers_empirical_claims() -> None:
    assert validate_source_index(load_valid_source_index(), load_valid_contract()) == []


def test_unindexed_claim_artifact_is_rejected() -> None:
    data = load_valid_contract()
    claim(data, "C5")["source_artifacts"].append("reports/not-indexed.csv")
    errors = validate_source_index(load_valid_source_index(), data)
    assert "C5 source artifact is not indexed: reports/not-indexed.csv" in errors


def test_invalid_source_sha256_is_rejected() -> None:
    sources = load_valid_source_index()
    sources["artifacts"][0]["source_sha256"] = "not-a-hash"
    errors = validate_source_index(sources, load_valid_contract())
    assert any("source_sha256 must be a lowercase SHA-256" in error for error in errors)


def test_duplicate_source_path_is_rejected() -> None:
    sources = load_valid_source_index()
    duplicate = deepcopy(sources["artifacts"][0])
    duplicate["id"] = "duplicate_fixture"
    sources["artifacts"].append(duplicate)
    errors = validate_source_index(sources, load_valid_contract())
    assert any("duplicate source artifact path" in error for error in errors)


def test_governance_documents_preserve_boundaries() -> None:
    provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    allowlist = (ROOT / "MIGRATION_ALLOWLIST.md").read_text(encoding="utf-8")
    assert validate_governance_documents(provenance, allowlist) == []


def test_missing_governance_section_is_rejected() -> None:
    provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    assert any(
        "MIGRATION_ALLOWLIST.md is missing" in error
        for error in validate_governance_documents(provenance, "")
    )
