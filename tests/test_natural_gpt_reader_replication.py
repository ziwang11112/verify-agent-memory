from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_natural_end_to_end_experiment as base
from scripts import run_natural_gpt_reader_replication as gpt


def test_gpt_reader_protocol_is_narrow_and_cost_capped() -> None:
    protocol = gpt.load_protocol()

    assert protocol.reader_binding.provider == "OpenAI"
    assert protocol.reader_binding.model == "gpt-5.6-luna"
    assert protocol.reader_binding.controls == {"effort": "none"}
    assert protocol.incremental_cap_usd == 45.0
    assert protocol.fixture_cap_usd == 0.02
    assert tuple(protocol.sample["arms"]) == gpt.PRIMARY_ARMS
    assert protocol.reporting["reader_estimates_pooled"] is False
    assert protocol.reporting["diagnostic_routes_not_rerun"] == [
        "namespace_text_verifier",
        "released_field_oracle",
    ]


def test_gpt_judge_protocol_is_cross_provider_and_exactly_bound() -> None:
    protocol = gpt.load_judge_protocol()

    assert protocol.judge["provider"] == "Anthropic"
    assert protocol.judge["model"] == "claude-haiku-4-5-20251001"
    assert protocol.judge["expected_unique_request_count"] == 3397
    assert protocol.judge["expected_request_id_set_sha256"] == (
        "5d4237295a917968ff95f18033cd75ba1f9b4707f1b008d16a7c3670d52fac07"
    )
    assert protocol.fixture_cap_usd == 0.01
    assert protocol.total_cap_usd == 10.0
    assert protocol.outputs["reader_estimates_pooled"] is False
    assert protocol.outputs["official_rhelm_or_memops_claim"] is False


def test_public_gpt_result_package_is_hash_bound_and_content_free() -> None:
    output = gpt.DEFAULT_OUTPUT
    manifest = base._read_json(output / "manifest.json")
    receipt = base._read_json(output / "execution_receipt.json")

    assert manifest["status"] == "complete_nonofficial_gpt_luna_primary_route_replication"
    assert manifest["reader_estimates_pooled"] is False
    assert manifest["official_rhelm_or_memops_claim"] is False
    assert manifest["sample_case_count"] == 1523
    assert manifest["unique_reader_request_count"] == 3157
    assert manifest["unique_judge_request_count"] == 3397
    for name, expected_hash in manifest["artifacts"].items():
        assert base._sha256_file(output / name) == expected_hash

    assert receipt["complete_bundle"] is True
    assert receipt["benchmark_payload_or_response_content_included"] is False
    assert receipt["official_benchmark_result"] is False
    assert receipt["combined_reader_and_judge_cost_usd"] == pytest.approx(21.778532)


def test_gpt_reader_plan_reuses_frozen_sample_and_only_primary_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = gpt.load_protocol()
    source = base.load_protocol(base.TWO_READER_PROTOCOL)
    selected = [SimpleNamespace(case_id="case-a"), SimpleNamespace(case_id="case-b")]
    assignments = {
        (case.case_id, arm): f"{case.case_id}-{arm}" for case in selected for arm in base.ARMS
    }
    specs = {
        request_id: base.CallSpec(
            request_id=request_id,
            payload=request_id,
            schema=base.reader_response_schema(),
        )
        for request_id in assignments.values()
    }
    primary_ids = {
        assignments[(case.case_id, arm)] for case in selected for arm in gpt.PRIMARY_ARMS
    }
    sample = {
        **protocol.sample,
        "case_count": len(selected),
        "assignment_count": len(selected) * len(gpt.PRIMARY_ARMS),
        "expected_unique_reader_request_count": len(primary_ids),
        "expected_reader_request_id_set_sha256": base._sha256_object(sorted(primary_ids)),
    }
    protocol = replace(protocol, sample=sample)
    monkeypatch.setattr(
        gpt,
        "_load_sources",
        lambda _protocol: (source, object(), selected, Path("unused")),
    )
    monkeypatch.setattr(
        base,
        "_reader_plan",
        lambda _cases, _protocol, _runtime: (specs, assignments, {}),
    )
    execution, selected, specs, assignments, _scores = gpt._reader_plan(protocol)

    assert len(selected) == 2
    assert len(assignments) == 6
    assert len(specs) == 6
    assert {arm for _case_id, arm in assignments} == set(gpt.PRIMARY_ARMS)
    assert [binding.provider for binding in execution.readers] == ["OpenAI"]
    assert base._sha256_object(sorted(specs)) == sample["expected_reader_request_id_set_sha256"]


def test_gpt_reader_fixture_uses_strict_schema_and_no_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    protocol = gpt.load_protocol()
    execution = gpt._execution_protocol(
        protocol,
        base.load_protocol(base.TWO_READER_PROTOCOL),
    )
    captured = {}

    def fake_provider(binding, **kwargs):
        captured["binding"] = binding
        captured.update(kwargs)
        return (
            {"action": "answer", "answer": "123 Market Street"},
            {"input_tokens": 10, "output_tokens": 8},
            1,
            1.0,
            "request-body-sha",
        )

    monkeypatch.setitem(base.provider_runtime.PROVIDER_CALLS, "OpenAI", fake_provider)
    monkeypatch.setattr(base, "_require_clean_contract", lambda: "test-commit")
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=test-only\n", encoding="ascii")

    receipt = base.run_fixture(
        protocol=execution,
        runtime=tmp_path / "runtime",
        stage="reader",
        binding=execution.readers[0],
        dotenv=dotenv,
    )

    assert captured["binding"].model == "gpt-5.6-luna"
    assert captured["binding"].controls == {"effort": "none"}
    assert captured["schema"] == base.reader_response_schema()
    assert receipt["synthetic_fixture"] is True
    assert receipt["cost_usd"] < protocol.fixture_cap_usd
    assert json.loads(json.dumps(receipt))["response"] is None
