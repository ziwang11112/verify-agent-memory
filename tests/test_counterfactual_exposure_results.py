from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.publish_counterfactual_exposure_results import validate_published

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "counterfactual_exposure"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: list[dict[str, str]], **selectors: str) -> dict[str, str]:
    matches = [
        row for row in rows if all(row.get(field) == value for field, value in selectors.items())
    ]
    assert len(matches) == 1
    return matches[0]


def test_published_counterfactual_exposure_bundle_passes() -> None:
    manifest = validate_published(RESULTS)

    assert manifest["execution_commit"] == "82d3bce8023d1ccc97bb21b0bbb36e15a4b3c6af"
    assert manifest["total_provider_calls"] == 1155
    assert manifest["model_pooling"] is False
    assert manifest["official_result"] is False


def test_published_bundle_contains_only_derived_fields() -> None:
    forbidden = {
        "answer",
        "answer_text",
        "prompt",
        "query_text",
        "response",
        "response_text",
        "target_marker",
    }
    for name in ("pair_scores.csv", "cell_metrics.csv", "bootstrap_ci.csv"):
        rows = _rows(name)
        assert rows
        assert not set(rows[0]).intersection(forbidden)


def test_headline_effects_match_frozen_scores() -> None:
    cells = _rows("cell_metrics.csv")
    intervals = _rows("bootstrap_ci.csv")
    gap = "relevant_admissible_minus_relevant_inadmissible_exposure_effect"

    expected = {
        "OpenAI": (0.90625, 0.0, 0.90625, 0.8125, 0.96875),
        "Gemini": (0.8125, 0.0, 0.8125, 0.6875, 0.9375),
        "DeepSeek": (0.8125, 0.15625, 0.65625, 0.5, 0.8125),
    }
    for provider, values in expected.items():
        admissible = _one(
            cells,
            provider=provider,
            scope="overall",
            cell="relevant_admissible",
        )
        inadmissible = _one(
            cells,
            provider=provider,
            scope="overall",
            cell="relevant_inadmissible",
        )
        selectivity = _one(
            intervals,
            provider=provider,
            scope="overall",
            contrast=gap,
        )
        observed = (
            float(admissible["exposure_effect"]),
            float(inadmissible["exposure_effect"]),
            float(selectivity["estimate"]),
            float(selectivity["ci_lower"]),
            float(selectivity["ci_upper"]),
        )
        assert observed == pytest.approx(values)


def test_deepseek_inadmissible_effect_interval_excludes_zero() -> None:
    row = _one(
        _rows("bootstrap_ci.csv"),
        provider="DeepSeek",
        scope="overall",
        contrast="relevant_inadmissible",
    )

    assert float(row["estimate"]) == pytest.approx(0.15625)
    assert float(row["ci_lower"]) == pytest.approx(0.03125)
    assert float(row["ci_upper"]) == pytest.approx(0.3125)


def test_provider_usage_records_no_retries_and_reconciles_cost() -> None:
    rows = _rows("provider_usage.csv")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 3
    assert all(int(row["total_calls"]) == 385 for row in rows)
    assert all(int(row["max_attempts"]) == 1 for row in rows)
    total = sum(float(row["total_cap_accounted_cost_usd"]) for row in rows)
    assert total == pytest.approx(manifest["total_cap_accounted_cost_usd"])
