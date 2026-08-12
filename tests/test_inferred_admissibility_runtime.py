from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_inferred_admissibility_experiment as runtime
from verify_agent_memory.inferred_admissibility import response_json_schema


def _prediction(candidate_count: int = 2) -> dict[str, object]:
    candidates = []
    for rank in range(1, candidate_count + 1):
        if rank == 1:
            policy = {"allowed": 0.8, "disallowed": 0.1, "unknown": 0.1}
            lifecycle = {"compatible": 0.1, "incompatible": 0.8, "unknown": 0.1}
            admissibility = {"admissible": 0.1, "inadmissible": 0.8, "unknown": 0.1}
        else:
            policy = {"allowed": 0.9, "disallowed": 0.05, "unknown": 0.05}
            lifecycle = {"compatible": 0.9, "incompatible": 0.05, "unknown": 0.05}
            admissibility = {"admissible": 0.9, "inadmissible": 0.05, "unknown": 0.05}
        candidates.append(
            {
                "candidate_key": f"c{rank:02d}",
                "policy": policy,
                "lifecycle": lifecycle,
                "admissibility": admissibility,
            }
        )
    return {
        "query_intent": {"current_state": 0.9, "history": 0.05, "unknown": 0.05},
        "candidates": candidates,
    }


def _binding(provider: str, *, controls: dict[str, str]) -> runtime.ProviderBinding:
    surfaces = {
        "OpenAI": "responses",
        "DeepSeek": "chat_completions",
        "Gemini": "generate_content",
        "Anthropic": "messages",
    }
    return runtime.ProviderBinding(
        provider=provider,
        model=f"fixture-{provider.lower()}",
        api_surface=surfaces[provider],
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        hard_cap_usd=1.0,
        controls=controls,
    )


def test_openai_adapter_uses_responses_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_http_json(**kwargs: object) -> tuple[dict[str, object], int, float]:
        captured.update(kwargs)
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(_prediction())}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        return response, 1, 12.0

    monkeypatch.setattr(runtime, "_http_json", fake_http_json)
    parsed, usage, attempts, _latency, _request_hash = runtime._openai_request(
        _binding("OpenAI", controls={"effort": "low"}),
        api_key="not-a-real-key",
        system_prompt="system",
        user_prompt="user",
        schema=response_json_schema(2),
        max_output_tokens=800,
        timeout_seconds=30,
        max_retries=0,
    )

    assert parsed == _prediction()
    assert usage == {"input_tokens": 10, "output_tokens": 20}
    assert attempts == 1
    assert captured["url"] == "https://api.openai.com/v1/responses"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("provider", ["DeepSeek", "Gemini", "Anthropic"])
def test_non_openai_adapters_parse_frozen_shapes(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    payload = json.dumps(_prediction())
    responses = {
        "DeepSeek": {
            "choices": [{"finish_reason": "stop", "message": {"content": payload}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 21},
        },
        "Gemini": {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": payload}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 22},
        },
        "Anthropic": {
            "content": [{"type": "text", "text": payload}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 13, "output_tokens": 23},
        },
    }

    def fake_http_json(**kwargs: object) -> tuple[dict[str, object], int, float]:
        captured.update(kwargs)
        return responses[provider], 1, 13.0

    monkeypatch.setattr(runtime, "_http_json", fake_http_json)
    controls = {
        "DeepSeek": {"thinking": "disabled"},
        "Gemini": {"thinking_level": "low"},
        "Anthropic": {"effort": "low"},
    }[provider]
    adapter = runtime.PROVIDER_CALLS[provider]
    parsed, usage, _attempts, _latency, _request_hash = adapter(
        _binding(provider, controls=controls),
        api_key="not-a-real-key",
        system_prompt="system JSON",
        user_prompt="user",
        schema=response_json_schema(2),
        max_output_tokens=800,
        timeout_seconds=30,
        max_retries=0,
    )

    assert parsed == _prediction()
    assert usage["input_tokens"] in {11, 12, 13}
    body = captured["body"]
    assert isinstance(body, dict)
    if provider == "DeepSeek":
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "disabled"}
    elif provider == "Gemini":
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
        sent_schema = body["generationConfig"]["responseJsonSchema"]
        sent_candidates = sent_schema["properties"]["candidates"]
        assert sent_candidates["minItems"] == 1
        assert "maxItems" not in sent_candidates
    else:
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["effort"] == "low"
        sent_schema = body["output_config"]["format"]["schema"]
        sent_candidates = sent_schema["properties"]["candidates"]
        assert sent_candidates["minItems"] == 1
        assert "maxItems" not in sent_candidates
        assert "exactly 2 items" in sent_candidates["description"]
        probability = sent_candidates["items"]["properties"]["policy"]["properties"]["allowed"]
        assert "minimum" not in probability
        assert "maximum" not in probability
        assert "between 0 and 1" in probability["description"]


def test_checkpoint_is_bound_to_exact_visible_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    binding = protocol.providers[0]
    case = runtime._provider_fixture_case()
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=not-a-real-key\n", encoding="utf-8")
    calls = 0

    def fake_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return _prediction(20), {"input_tokens": 10, "output_tokens": 20}, 1, 5.0, "request"

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", fake_provider)
    first = runtime.execute_provider(
        protocol,
        (case,),
        binding=binding,
        dotenv=dotenv,
        output_dir=tmp_path,
    )
    second = runtime.execute_provider(
        protocol,
        (case,),
        binding=binding,
        dotenv=dotenv,
        output_dir=tmp_path,
    )

    assert first["completed_cases"] == 1
    assert second["completed_cases"] == 1
    assert calls == 1

    changed = replace(case, query_text="Where is the replacement card going today?")
    with pytest.raises(ValueError, match="case payload binding drifted"):
        runtime.execute_provider(
            protocol,
            (changed,),
            binding=binding,
            dotenv=dotenv,
            output_dir=tmp_path,
        )


def test_provider_failure_marker_records_content_free_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    binding = protocol.providers[0]
    case = runtime._provider_fixture_case()
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=not-a-real-key\n", encoding="utf-8")

    def invalid_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise ValueError("prediction candidate count differs from request")

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", invalid_provider)
    with pytest.raises(ValueError, match="candidate count"):
        runtime.execute_provider(
            protocol,
            (case,),
            binding=binding,
            dotenv=dotenv,
            output_dir=tmp_path,
        )

    failure = json.loads(runtime._failure_path(tmp_path, binding).read_text(encoding="utf-8"))
    assert failure["error_type"] == "ValueError"
    assert failure["error_message"] == "prediction candidate count differs from request"
    assert failure["semantic_or_output_repair_attempted"] is False


def test_duplicate_checkpoint_recovery_is_first_committed_with_last_sensitivity(
    tmp_path: Path,
) -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    binding = protocol.providers[0]
    case = runtime._provider_fixture_case()
    source_dir = tmp_path / "raw"
    source_path = runtime._response_path(source_dir, binding)
    base = {
        "case_id": case.case_id,
        "provider": binding.provider,
        "model": binding.model,
        "input_tokens": 10,
        "output_tokens": 20,
        "cost_usd": 0.001,
        "attempts": 1,
        "latency_ms": 5.0,
        "request_sha256": "same-request",
        "protocol_sha256": runtime.sha256_file(protocol.path),
        "prompt_sha256": protocol.prompt_sha256,
        "case_payload_sha256": runtime._sha256_text(runtime.prompt_payload(case)),
        "raw_query_or_memory_text_saved": False,
    }
    first = _prediction(20)
    last = _prediction(20)
    last["query_intent"] = {"current_state": 0.8, "history": 0.1, "unknown": 0.1}
    runtime._append_jsonl(source_path, {**base, "prediction": first})
    runtime._append_jsonl(source_path, {**base, "prediction": last})
    source_hash = runtime.sha256_file(source_path)

    output_dir = tmp_path / "canonical"
    audit = runtime.canonicalize_response_checkpoints(
        protocol,
        (case,),
        bindings=(binding,),
        response_dir=source_dir,
        output_dir=output_dir,
    )

    primary = runtime._load_response_records(
        runtime._response_path(output_dir / "primary_first_committed", binding)
    )
    sensitivity = runtime._load_response_records(
        runtime._response_path(output_dir / "sensitivity_last_committed", binding)
    )
    assert primary[0]["prediction"] == first
    assert sensitivity[0]["prediction"] == last
    assert runtime.sha256_file(source_path) == source_hash
    provider_audit = audit["providers"][0]
    assert provider_audit["duplicate_case_groups"] == 1
    assert provider_audit["duplicate_calls"] == 1
    assert provider_audit["duplicate_groups_with_different_predictions"] == 1
    assert audit["raw_checkpoints_modified"] is False


def test_score_pipeline_writes_all_content_free_outputs(tmp_path: Path) -> None:
    base_protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    raw = dict(base_protocol.raw)
    evaluation = dict(raw["evaluation"])
    evaluation["bootstrap_replicates"] = 20
    raw["evaluation"] = evaluation
    protocol = replace(base_protocol, raw=raw)
    base = runtime._provider_fixture_case()
    cases = []
    for source, intent in (
        ("rhelm", "current_state"),
        ("memops", "current_state"),
        ("memops", "history"),
    ):
        for role in ("calibration", "analysis"):
            cases.append(
                replace(
                    base,
                    case_id=f"{source}-{intent}-{role}",
                    source=source,
                    group_id=f"group-{source}-{intent}",
                    role=role,
                    released_query_intent=intent,
                )
            )

    response_dir = tmp_path / "responses"
    protocol_hash = runtime.sha256_file(protocol.path)
    for binding in protocol.providers:
        path = runtime._response_path(response_dir, binding)
        for case in cases:
            prediction = _prediction(20)
            if case.released_query_intent == "history":
                prediction["query_intent"] = {
                    "current_state": 0.05,
                    "history": 0.9,
                    "unknown": 0.05,
                }
            runtime._append_jsonl(
                path,
                {
                    "case_id": case.case_id,
                    "provider": binding.provider,
                    "model": binding.model,
                    "prediction": prediction,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cost_usd": 0.001,
                    "attempts": 1,
                    "latency_ms": 5.0,
                    "request_sha256": "request",
                    "protocol_sha256": protocol_hash,
                    "prompt_sha256": protocol.prompt_sha256,
                    "case_payload_sha256": runtime._sha256_text(runtime.prompt_payload(case)),
                    "raw_query_or_memory_text_saved": False,
                },
            )

    output_dir = tmp_path / "scores"
    manifest = runtime.score_responses(
        protocol,
        tuple(cases),
        response_dir=response_dir,
        output_dir=output_dir,
    )

    expected = {
        "selected_thresholds.csv",
        "classification_metrics.csv",
        "filter_metrics.csv",
        "route_metrics.csv",
        "paired_route_deltas.csv",
        "provider_usage.csv",
    }
    assert set(manifest["outputs"]) == expected
    with (output_dir / "paired_route_deltas.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 80
    assert any(
        row["arm"] == "released_oracle" and row["comparator"] == "namespace_dense" for row in paired
    )
    assert {row["bootstrap_stratification"] for row in paired} == {"source_query_intent"}

    subset_dir = tmp_path / "subset-scores"
    subset = runtime.score_responses(
        protocol,
        tuple(cases),
        response_dir=response_dir,
        output_dir=subset_dir,
        bindings=(protocol.providers[0], protocol.providers[2]),
    )
    assert subset["scored_complete_providers"] == ["OpenAI", "Gemini"]
    assert subset["excluded_providers"] == ["DeepSeek", "Anthropic"]
