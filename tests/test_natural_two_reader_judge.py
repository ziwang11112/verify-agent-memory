from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_natural_end_to_end_experiment as base
from scripts import run_natural_two_reader_judge as judge


def _protocol_for_cases(cases: list[SimpleNamespace]) -> judge.JudgeProtocol:
    rhelm = [case for case in cases if case.source == "rhelm"]
    memops = [case for case in cases if case.source == "memops"]
    salt = "test-judge|"
    selected_memops = sorted(
        memops,
        key=lambda case: (
            hashlib.sha256(f"{salt}{case.case_id}".encode()).hexdigest(),
            case.case_id,
        ),
    )[:2]
    selected = sorted([*rhelm, *selected_memops], key=lambda case: case.case_id)
    sampling = {
        "rhelm": {"population_count": len(rhelm)},
        "memops": {
            "population_count": len(memops),
            "sample_count": 2,
            "salt": salt,
            "sample_namespace_group_count": len({case.group_id for case in selected_memops}),
        },
        "sample_case_count": len(selected),
        "sample_case_id_set_sha256": base._sha256_object([case.case_id for case in selected]),
    }
    return judge.JudgeProtocol(
        raw={},
        path=judge.DEFAULT_JUDGE_PROTOCOL,
        sha256="protocol-sha",
        source={},
        gate={},
        sampling=sampling,
        judge={},
        budget={
            "fixture_hard_cap_usd": 0.01,
            "planned_calls_conservative_cap_usd": 45.0,
            "total_incremental_hard_cap_usd": 60.0,
        },
        outputs={},
    )


def test_frozen_sampled_judge_protocol_loads() -> None:
    protocol = judge.load_judge_protocol(judge.DEFAULT_JUDGE_PROTOCOL)

    assert protocol.fixture_cap_usd == 0.01
    assert protocol.plan_cap_usd == 45.0
    assert protocol.total_cap_usd == 60.0
    assert protocol.sampling["outcome_or_route_dependent_selection"] is False
    assert protocol.sampling["arms"] == list(base.ARMS)
    assert protocol.judge["expected_unique_request_count"] == 8790
    assert protocol.judge["model"] == "claude-haiku-4-5-20251001"
    assert protocol.judge["effort"] == "omitted"


def test_case_sampling_is_deterministic_and_ignores_outcomes() -> None:
    cases = [
        SimpleNamespace(case_id="r1", source="rhelm", group_id="rg", answer="wrong"),
        SimpleNamespace(case_id="r2", source="rhelm", group_id="rg", answer="right"),
        SimpleNamespace(case_id="m1", source="memops", group_id="g1", answer="a"),
        SimpleNamespace(case_id="m2", source="memops", group_id="g2", answer="b"),
        SimpleNamespace(case_id="m3", source="memops", group_id="g3", answer="c"),
        SimpleNamespace(case_id="m4", source="memops", group_id="g4", answer="d"),
    ]
    protocol = _protocol_for_cases(cases)
    first = judge.select_cases(cases, protocol)

    for case in cases:
        case.answer = "changed-after-freeze"
    second = judge.select_cases(list(reversed(cases)), protocol)

    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert {case.case_id for case in first if case.source == "rhelm"} == {"r1", "r2"}
    assert sum(case.source == "memops" for case in first) == 2


def test_fixture_plan_uses_the_frozen_judge_schema() -> None:
    protocol = judge.load_judge_protocol(judge.DEFAULT_JUDGE_PROTOCOL)
    execution = judge.bind_recovered_judge(
        protocol,
        base.load_protocol(judge.DEFAULT_EXECUTION_PROTOCOL),
    )
    spec = judge._fixture_spec(execution)

    assert execution.judge.model == "claude-haiku-4-5-20251001"
    assert execution.judge.controls == {"thinking": "disabled", "effort": "omitted"}
    assert spec.schema == base.judge_response_schema()
    assert spec.request_id == "synthetic-natural-end-to-end-fixture"
    assert "global_dense" not in spec.payload
    assert "Gemini" not in spec.payload


def test_total_budget_reserves_fixture_cost() -> None:
    protocol = judge.load_judge_protocol(judge.DEFAULT_JUDGE_PROTOCOL)
    fixture_cost = 0.004

    assert protocol.total_cap_usd - fixture_cost == 59.996


def test_recovered_haiku_adapter_omits_only_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_http_json(**kwargs: object):
        captured.update(kwargs)
        return (
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "answer_correct": True,
                                "answer_quality": 9,
                                "protected_disclosure": False,
                                "stale_disclosure": False,
                                "reason": "matches",
                            }
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
            1,
            12.0,
        )

    monkeypatch.setattr(base.provider_runtime, "_http_json", fake_http_json)
    protocol = judge.load_judge_protocol(judge.DEFAULT_JUDGE_PROTOCOL)
    execution = judge.bind_recovered_judge(
        protocol,
        base.load_protocol(judge.DEFAULT_EXECUTION_PROTOCOL),
    )
    response, *_ = judge._anthropic_haiku_request(
        execution.judge.runtime_binding(),
        api_key="not-a-real-key",
        system_prompt="judge",
        user_prompt="payload",
        schema=base.judge_response_schema(),
        max_output_tokens=256,
        timeout_seconds=30,
        max_retries=0,
    )

    assert response["answer_correct"] is True
    body = captured["body"]
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["thinking"] == {"type": "disabled"}
    assert "effort" not in body["output_config"]
    assert body["output_config"]["format"]["type"] == "json_schema"
