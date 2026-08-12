from __future__ import annotations

from pathlib import Path

from scripts.import_posthoc_evidence import (
    _controlled_rows,
    _inferred_rows,
    _metadata_reliability_rows,
    _natural_fixed_budget_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_budget_import_is_posthoc_and_complete() -> None:
    rows = _natural_fixed_budget_rows(ROOT / "results" / "supplemental_natural")
    assert len(rows) == 44
    assert {row["contrast"] for row in rows} == {
        "top_k_10",
        "top_k_20",
        "top_k_50",
        "top_k_100",
    }
    assert all("post_hoc_v2" in row["notes"] and "no_retuning" in row["notes"] for row in rows)


def test_metadata_import_keeps_grid_and_reranking_boundaries() -> None:
    rows = _metadata_reliability_rows(ROOT / "results" / "supplemental_natural")
    assert len(rows) == 37
    swap = next(
        row
        for row in rows
        if row["contrast"] == "namespace_swap"
        and row["metric"] == "first_observed_non_dominating_rate"
    )
    assert swap["estimate"] == "0.2"
    assert "observed_grid_bracket" in swap["notes"]
    full_reranking = next(row for row in rows if row["contrast"] == "false_allow_at_0_5")
    assert "full_natural_reranking" in full_reranking["notes"]
    assert "no_retuning" in full_reranking["notes"]


def test_inferred_import_never_pools_models() -> None:
    rows = _inferred_rows(ROOT / "results" / "inferred_admissibility")
    assert len(rows) == 17
    assert all("no_model_pooling" in row["notes"] for row in rows)
    assert {row["contrast"] for row in rows} == {
        "released_oracle",
        "openai_text_inferred",
        "gemini_text_inferred",
    }


def test_controlled_import_accepts_ci_lower_schema() -> None:
    rows = _controlled_rows(ROOT / "results" / "counterfactual_admissibility")
    assert len(rows) == 12
    openai_overflip = next(
        row
        for row in rows
        if row["contrast"] == "gpt56_controlled" and row["metric"] == "stable_control_overflip_rate"
    )
    assert openai_overflip["ci95_lower"] == "0.109375"
    assert openai_overflip["ci95_upper"] == "0.3125"
