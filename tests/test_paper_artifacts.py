from __future__ import annotations

from pathlib import Path

from scripts.build_paper_artifacts import (
    NATURAL_ARMS,
    _evidence_rows,
    _load_figure_examples,
    _load_rows,
)
from scripts.verify_paper import validate_paper

ROOT = Path(__file__).resolve().parents[1]


def test_paper_evidence_selectors_cover_primary_claims() -> None:
    selected = _evidence_rows(_load_rows(ROOT))
    assert selected["reader_a"]["claim_id"] == "C2"
    assert selected["reader_a_brier"]["metric"] == "brier_delta"
    assert selected["g1_leakage"]["claim_id"] == "C3"
    assert selected["smoke_global_recall"]["claim_id"] == "C4"
    assert selected["namespace_recall_delta"]["claim_id"] == "C5"
    assert selected["namespace_wrong_scope"]["estimate"] == "0.0"
    assert selected["global_candidates"]["metric"] == "mean_candidates_scored"
    assert selected["namespace_candidates"]["metric"] == "mean_candidates_scored"
    assert selected["global_penalized_risk"]["metric"] == "penalized_non_usable_upper_risk"
    assert selected["namespace_label_coverage"]["contrast"] == "namespace_dense_matched_prefix"
    assert selected["threshold_recall_delta"]["claim_id"] == "C6"
    assert selected["threshold_fallback_rate"]["metric"] == "fallback_rate"
    assert selected["lifecycle_contamination_delta"]["claim_id"] == "C7"
    assert selected["c13_deepseek_namespace_answer_correct"]["claim_id"] == "C13"
    assert selected["c13_gemini_namespace_answer_correct"]["claim_id"] == "C13"
    assert selected["c13_gpt_luna_namespace_answer_correct"]["claim_id"] == "C13"
    assert selected["c14_overall_exact_agreement"]["claim_id"] == "C14"
    assert selected["c14_original_two_reader_exact_agreement"]["n"] == "134"
    assert selected["c14_sequential_gpt_reader_exact_agreement"]["n"] == "66"


def test_natural_end_to_end_table_uses_claim_bound_generated_values() -> None:
    selected = _evidence_rows(_load_rows(ROOT))
    assert selected["c13_deepseek_namespace_answer_quality"]["claim_id"] == "C13"
    assert selected["c13_gemini_namespace_over_refusal"]["claim_id"] == "C13"
    assert selected["c13_gpt_luna_policy_answer_correct"]["claim_id"] == "C13"

    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for macro in (
        r"\NaturalEndToEndRecallDelta",
        r"\NaturalEndToEndAdmRiskDelta",
        r"\NaturalPolicyAdmRiskDelta",
        r"\NaturalTextVerifierAdmRiskDelta",
        r"\DeepSeekNaturalAccuracyDelta",
        r"\GeminiNaturalAccuracyDelta",
        r"\GPTLunaNaturalAccuracyDelta",
        r"\CrossJudgeAgreement",
        r"\CrossJudgeAgreementCI",
    ):
        assert macro in main
    assert r"\input{generated/natural_end_to_end_results.tex}" in main


def test_paper_package_passes() -> None:
    assert validate_paper(ROOT) == []


def test_generated_tex_uses_platform_independent_line_endings() -> None:
    for path in sorted((ROOT / "paper" / "generated").glob("*.tex")):
        assert b"\r\n" not in path.read_bytes()


def test_generated_tables_keep_v1_non_usable_risk_distinct() -> None:
    generated = ROOT / "paper" / "generated"
    main_results = (generated / "main_results.tex").read_text(encoding="ascii")
    full_arms = (generated / "full_arm_results.tex").read_text(encoding="ascii")
    matched_prefix = (generated / "matched_prefix_diagnostics.tex").read_text(encoding="ascii")
    smoke = (generated / "mechanism_smoke.tex").read_text(encoding="ascii")

    assert "Penalized conservative risk" not in main_results
    assert "Lifecycle upper bound" not in main_results
    assert "Penalized non-usable upper risk" in main_results
    assert "Historical released-field v1" in main_results
    assert "Penalized non-usable" in full_arms
    assert "Known non-usable" in matched_prefix
    assert "Non-usable fraction" in smoke


def test_retrieval_figure_covers_all_frozen_natural_arms() -> None:
    assert len(NATURAL_ARMS) == 9


def test_figure_one_examples_cover_distinct_admissibility_cases() -> None:
    cases = _load_figure_examples(ROOT)
    assert {case["case_id"] for case in cases} == {
        "wrong_namespace_market",
        "superseded_lisbon_date",
        "forgotten_phone_detail",
        "historical_metformin_timeline",
    }
    assert any(
        memory["scope"] == "disallowed" and memory["usable"] is False
        for case in cases
        for memory in case["memories"]
    )
    assert any(
        memory["state"] == "superseded" and memory["usable"] is False
        for case in cases
        for memory in case["memories"]
    )
    assert any(
        memory["prohibited"] is True and memory["usable"] is False
        for case in cases
        for memory in case["memories"]
    )
    assert any(
        memory["state"] == "superseded" and memory["usable"] is True
        for case in cases
        for memory in case["memories"]
    )


def test_generated_svg_uses_platform_independent_line_endings() -> None:
    for path in sorted((ROOT / "paper" / "generated").glob("*.svg")):
        assert b"\r\n" not in path.read_bytes()
