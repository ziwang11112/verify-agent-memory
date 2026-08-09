from __future__ import annotations

import json

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


def test_gpt_reader_plan_reuses_frozen_sample_and_only_primary_routes() -> None:
    protocol = gpt.load_protocol()
    execution, selected, specs, assignments, _scores = gpt._reader_plan(protocol)

    assert len(selected) == 1523
    assert len(assignments) == 4569
    assert len(specs) == 3157
    assert {arm for _case_id, arm in assignments} == set(gpt.PRIMARY_ARMS)
    assert [binding.provider for binding in execution.readers] == ["OpenAI"]
    assert base._sha256_object(sorted(specs)) == (
        "c119f0b6a243adb0e684f0368983c638c3cdfa6c99d4106ffa7f3d919933f7f2"
    )


def test_gpt_reader_fixture_uses_strict_schema_and_no_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    protocol = gpt.load_protocol()
    execution, *_ = gpt._reader_plan(protocol)
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
