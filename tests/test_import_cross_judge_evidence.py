from __future__ import annotations

from pathlib import Path

from scripts.import_cross_judge_evidence import normalized_rows

ROOT = Path(__file__).resolve().parents[1]


def test_cross_judge_import_preserves_bounded_audit_scope() -> None:
    rows = normalized_rows(ROOT / "results" / "natural_cross_judge_audit")

    assert len(rows) == 16
    assert all(row["claim_id"] == "C14" for row in rows)
    assert all("post_hoc_outcome_independent" in str(row["notes"]) for row in rows)
    assert all("full_population_not_rescored" in str(row["notes"]) for row in rows)

    keyed = {(row["contrast"], row["metric"]): row for row in rows}
    assert keyed[("overall", "exact_agreement")]["estimate"] == "0.865"
    assert keyed[("overall", "exact_agreement")]["ci95_lower"] == "0.815"
    assert keyed[("original_two_reader", "exact_agreement")]["n"] == "134"
    assert keyed[("sequential_gpt_reader", "exact_agreement")]["estimate"] == ("0.8333333333333334")
    assert keyed[("directional_disagreement", "cheap_correct_strong_incorrect")]["estimate"] == "14"
    assert keyed[("directional_disagreement", "cheap_incorrect_strong_correct")]["estimate"] == "13"
