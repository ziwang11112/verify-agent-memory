from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_counterfactual_admissibility_experiment as runtime


def _all_admissible_response(candidate_count: int = 3) -> dict[str, object]:
    return {
        "candidates": [
            {
                "candidate_key": f"m{index}",
                "admissibility": {
                    "admissible": 0.9,
                    "inadmissible": 0.05,
                    "unknown": 0.05,
                },
            }
            for index in range(1, candidate_count + 1)
        ]
    }


def _dotenv(path: Path, provider: str = "OpenAI") -> Path:
    key = {
        "OpenAI": "OPENAI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    path.write_text(f"{key}=not-a-real-key\n", encoding="utf-8")
    return path


def test_protocol_is_hash_bound_and_balanced() -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    receipt = runtime.validate_protocol(protocol)

    assert receipt["scenario_count"] == 16
    assert receipt["pair_count"] == 32
    assert receipt["case_count"] == 64
    assert receipt["candidate_judgment_count"] == 192
    assert receipt["official_result"] is False


def test_fixture_receipt_is_bound_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    binding = runtime._binding(protocol, "OpenAI")
    calls = 0

    def fake_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return (
            _all_admissible_response(),
            {"input_tokens": 10, "output_tokens": 20},
            1,
            5.0,
            "request",
        )

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", fake_provider)
    dotenv = _dotenv(tmp_path / ".env")
    first = runtime.fixture_provider(
        protocol,
        binding=binding,
        dotenv=dotenv,
        runtime_dir=tmp_path / "runtime",
    )
    second = runtime.fixture_provider(
        protocol,
        binding=binding,
        dotenv=dotenv,
        runtime_dir=tmp_path / "runtime",
    )

    assert first == second
    assert calls == 1
    assert first["prediction_saved"] is False
    assert first["query_or_candidate_text_saved"] is False


def test_complete_checkpoint_scores_and_publishes_without_raw_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    raw = dict(base.raw)
    evaluation = dict(raw["evaluation"])
    evaluation["bootstrap_replicates"] = 20
    raw["evaluation"] = evaluation
    protocol = replace(base, raw=raw)
    binding = runtime._binding(protocol, "OpenAI")
    calls = 0

    def fake_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return (
            _all_admissible_response(),
            {"input_tokens": 10, "output_tokens": 20},
            1,
            5.0,
            "request",
        )

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", fake_provider)
    dotenv = _dotenv(tmp_path / ".env")
    runtime_dir = tmp_path / "runtime"
    runtime.fixture_provider(
        protocol,
        binding=binding,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )
    completion = runtime.execute_provider(
        protocol,
        binding=binding,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )
    resumed = runtime.execute_provider(
        protocol,
        binding=binding,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )

    assert completion == resumed
    assert completion["completed_cases"] == 64
    assert calls == 65

    score_dir = tmp_path / "scores"
    manifest = runtime.score_responses(
        protocol,
        runtime_dir=runtime_dir,
        score_dir=score_dir,
    )
    assert manifest["comparison_eligible_providers"] == [
        {"provider": "OpenAI", "model": binding.model}
    ]
    assert len(manifest["excluded_providers"]) == 3

    publish_dir = tmp_path / "published"
    publication = runtime.publish_scores(
        protocol,
        score_dir=score_dir,
        publish_dir=publish_dir,
    )
    assert publication["comparison_eligible_provider_count"] == 1
    assert (publish_dir / "README.md").is_file()
    assert not (publish_dir / "raw").exists()
    published_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in publish_dir.iterdir()
        if path.suffix in {".md", ".json", ".csv"}
    )
    assert "not-a-real-key" not in published_text
    assert "Scope: Monica only" not in published_text


def test_invalid_provider_output_freezes_without_automatic_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_protocol(runtime.DEFAULT_PROTOCOL)
    binding = runtime._binding(protocol, "OpenAI")
    dotenv = _dotenv(tmp_path / ".env")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(runtime, "_adapter_sha256", lambda _provider: "test-adapter")

    def valid_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        return (
            _all_admissible_response(),
            {"input_tokens": 10, "output_tokens": 20},
            1,
            5.0,
            "request",
        )

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", valid_provider)
    runtime.fixture_provider(
        protocol,
        binding=binding,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )

    calls = 0

    def invalid_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return {"candidates": []}, {"input_tokens": 10, "output_tokens": 2}, 1, 5.0, "request"

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", invalid_provider)
    with pytest.raises(ValueError, match="candidate count"):
        runtime.execute_provider(
            protocol,
            binding=binding,
            dotenv=dotenv,
            runtime_dir=runtime_dir,
        )
    failure = json.loads(runtime._failure_path(runtime_dir, binding).read_text(encoding="utf-8"))
    assert failure["status"] == "frozen_failure"
    assert failure["semantic_or_output_repair_attempted"] is False
    assert failure["automatic_rerun_allowed"] is False

    with pytest.raises(RuntimeError, match="frozen failure marker"):
        runtime.execute_provider(
            protocol,
            binding=binding,
            dotenv=dotenv,
            runtime_dir=runtime_dir,
        )
    assert calls == 1
