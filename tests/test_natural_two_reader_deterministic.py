from __future__ import annotations

from scripts import analyze_natural_two_reader_deterministic as analysis


def _paired(provider: str, arm: str, metric: str, value: float) -> dict[str, object]:
    return {
        "reader_provider": provider,
        "arm": arm,
        "metric": metric,
        "mean_delta_arm_minus_reference": value,
    }


def _gate_rows(
    *,
    gemini_utility: float = -0.01,
    deepseek_utility: float = -0.02,
    risk: float = -0.1,
) -> list[dict[str, object]]:
    rows = []
    for provider, utility in (("Gemini", gemini_utility), ("DeepSeek", deepseek_utility)):
        for arm in ("namespace_dense", "namespace_policy_gate"):
            rows.extend(
                [
                    _paired(provider, arm, "wrong_namespace_exposure", -0.2),
                    _paired(
                        provider,
                        arm,
                        "penalized_admissibility_upper_risk",
                        risk,
                    ),
                    _paired(provider, arm, "contains_reference_answer", utility),
                    _paired(provider, arm, "over_refusal", 0.01),
                ]
            )
    return rows


def test_frozen_deterministic_analysis_protocol_loads() -> None:
    protocol = analysis.load_analysis_protocol(analysis.DEFAULT_ANALYSIS_PROTOCOL)
    assert protocol["provider_calls_authorized"] == 0
    assert protocol["comparison"]["reader_pooling"] is False
    assert protocol["judge_continuation_gate"]["all_conditions_required"] is True


def test_deterministic_gate_passes_only_with_route_and_each_reader_utility() -> None:
    protocol = analysis.load_analysis_protocol(analysis.DEFAULT_ANALYSIS_PROTOCOL)
    passed = analysis._evaluate_gate(protocol, _gate_rows())
    assert passed["passed"] is True

    bad_reader = analysis._evaluate_gate(
        protocol,
        _gate_rows(deepseek_utility=-0.051),
    )
    assert bad_reader["route_gate_passed"] is True
    assert bad_reader["reader_gate_passed"] is False
    assert bad_reader["passed"] is False

    bad_route = analysis._evaluate_gate(protocol, _gate_rows(risk=0.0))
    assert bad_route["route_gate_passed"] is False
    assert bad_route["passed"] is False
