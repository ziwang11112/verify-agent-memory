from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "counterfactual_admissibility"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_binds_all_generated_outputs() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == ("controlled_public_dev_diagnostic_not_official_benchmark_result")
    assert manifest["official_result"] is False
    assert manifest["prefix_scoring"] is False
    assert manifest["selection_or_tuning"] is False
    assert manifest["comparison_eligible_providers"] == [
        {"model": "gpt-5.6-sol", "provider": "OpenAI"},
        {"model": "deepseek-v4-pro", "provider": "DeepSeek"},
        {"model": "gemini-3.6-flash", "provider": "Gemini"},
    ]
    assert manifest["excluded_providers"] == [
        {
            "model": "claude-sonnet-5",
            "provider": "Anthropic",
            "reason": "frozen_failure",
        }
    ]
    for output in manifest["outputs"]:
        path = RESULTS / output["path"]
        assert path.is_file()
        assert _sha256(path) == output["sha256"]


def test_primary_result_preserves_focal_and_control_outcomes() -> None:
    rows = {
        row["model"]: row
        for row in _csv(RESULTS / "aggregate_metrics.csv")
        if row["scope"] == "overall"
    }

    assert float(rows["OpenAI/gpt-5.6-sol"]["strict_focal_pair_consistency"]) == 1
    assert float(rows["Gemini/gemini-3.6-flash"]["strict_focal_pair_consistency"]) == 1
    assert float(
        rows["DeepSeek/deepseek-v4-pro"]["strict_focal_pair_consistency"]
    ) == pytest.approx(0.71875)
    assert float(rows["OpenAI/gpt-5.6-sol"]["stable_control_overflip_rate"]) == pytest.approx(
        0.203125
    )
    assert float(rows["DeepSeek/deepseek-v4-pro"]["stable_control_overflip_rate"]) == pytest.approx(
        0.265625
    )
    assert float(rows["Gemini/gemini-3.6-flash"]["stable_control_overflip_rate"]) == pytest.approx(
        0.296875
    )


def test_no_provider_meets_the_frozen_complete_pattern() -> None:
    rows = _csv(RESULTS / "aggregate_metrics.csv")
    providers = {
        "OpenAI/gpt-5.6-sol",
        "DeepSeek/deepseek-v4-pro",
        "Gemini/gemini-3.6-flash",
    }
    overall = {row["model"]: row for row in rows if row["scope"] == "overall"}
    axes = {
        model: [row for row in rows if row["scope"] == "axis" and row["model"] == model]
        for model in providers
    }

    for model in providers:
        supportive = (
            float(overall[model]["strict_focal_pair_consistency"]) >= 0.8
            and all(float(row["strict_focal_pair_consistency"]) >= 0.75 for row in axes[model])
            and float(overall[model]["stable_control_overflip_rate"]) <= 0.05
            and float(overall[model]["mean_focal_directional_margin"]) > 0
        )
        assert supportive is False


def test_published_bundle_contains_no_raw_prompt_or_response_text() -> None:
    forbidden = (
        "Scope: Monica only",
        "not-a-real-key",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
    )
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in RESULTS.iterdir()
        if path.suffix in {".csv", ".json", ".md"}
    )
    assert all(value not in text for value in forbidden)


def test_posthoc_taxonomy_is_zero_call_and_hash_bound() -> None:
    manifest = json.loads((RESULTS / "posthoc_manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "posthoc_descriptive_error_taxonomy"
    assert manifest["comparison_eligible_provider_count"] == 3
    assert manifest["provider_call_count"] == 0
    assert manifest["prompt_or_prediction_modified"] is False
    assert manifest["individual_case_rows_published"] is False
    assert manifest["query_or_candidate_text_published"] is False
    assert manifest["official_result"] is False
    assert len(manifest["checkpoints"]) == 3
    assert all(checkpoint["case_count"] == 64 for checkpoint in manifest["checkpoints"])
    for output in manifest["outputs"]:
        path = RESULTS / output["path"]
        assert path.is_file()
        assert _sha256(path) == output["sha256"]


def test_posthoc_taxonomy_identifies_false_denial_as_dominant_error() -> None:
    rows = _csv(RESULTS / "error_taxonomy.csv")
    overall = {
        (row["model"], row["role"]): row
        for row in rows
        if row["axis"] == "all" and row["condition"] == "all"
    }
    expected = {
        "OpenAI/gpt-5.6-sol": (0.265625, 0.03125),
        "DeepSeek/deepseek-v4-pro": (0.59375, 0.015625),
        "Gemini/gemini-3.6-flash": (0.53125, 0.0),
    }
    for model, (false_deny, false_admit) in expected.items():
        assert float(overall[(model, "stable_admissible")]["false_deny_rate"]) == pytest.approx(
            false_deny
        )
        assert float(overall[(model, "stable_inadmissible")]["false_admit_rate"]) == pytest.approx(
            false_admit
        )
        assert false_deny > false_admit


def test_posthoc_outputs_publish_no_individual_identifiers() -> None:
    for name in (
        "error_taxonomy.csv",
        "stable_control_outcomes.csv",
        "overflip_transitions.csv",
    ):
        rows = _csv(RESULTS / name)
        assert rows
        assert "pair_id" not in rows[0]
        assert "scenario_id" not in rows[0]
        assert "query" not in rows[0]
        assert "text" not in rows[0]
