from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.publish_claude_opus5_exposure_results import validate_published

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "claude_opus5_exposure_replication"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: list[dict[str, str]], **selectors: str) -> dict[str, str]:
    matches = [
        row for row in rows if all(row.get(field) == value for field, value in selectors.items())
    ]
    assert len(matches) == 1
    return matches[0]


def test_opus_replication_bundle_passes_and_remains_separate() -> None:
    manifest = validate_published(RESULTS)

    assert manifest["execution_commit"] == "938320909b4c2d13e286987dc4297a7cb6ef73a7"
    assert manifest["comparison_eligible_provider"] == {
        "provider": "Anthropic",
        "model": "claude-opus-5",
    }
    assert manifest["relation_to_frozen_panel"] == "separate_fourth_reader_replication_no_pooling"
    assert manifest["model_pooling"] is False
    assert manifest["official_result"] is False


def test_opus_replication_headline_effects_match_frozen_scores() -> None:
    cells = _rows("cell_metrics.csv")
    intervals = _rows("bootstrap_ci.csv")
    admissible = _one(cells, scope="overall", cell="relevant_admissible")
    inadmissible = _one(cells, scope="overall", cell="relevant_inadmissible")
    gap = _one(
        intervals,
        scope="overall",
        contrast="relevant_admissible_minus_relevant_inadmissible_exposure_effect",
    )

    assert float(admissible["exposure_effect"]) == pytest.approx(0.96875)
    assert float(inadmissible["exposure_effect"]) == pytest.approx(0.125)
    assert (
        float(gap["estimate"]),
        float(gap["ci_lower"]),
        float(gap["ci_upper"]),
    ) == pytest.approx((0.84375, 0.6875, 0.96875))


def test_opus_replication_contains_no_content_fields_and_reconciles_cost() -> None:
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

    usage = _one(_rows("provider_usage.csv"), provider="Anthropic")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert int(usage["total_calls"]) == 385
    assert int(usage["max_attempts"]) == 1
    assert float(usage["total_cap_accounted_cost_usd"]) == pytest.approx(
        manifest["total_cap_accounted_cost_usd"]
    )
