from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_counterfactual_exposure_execution as runtime


def _unlock(
    protocol: runtime.ExecutionProtocol,
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(runtime, "_head_commit", lambda: "a" * 40)
    monkeypatch.setattr(runtime, "_contract_dirty_paths", lambda: ())
    path.write_text(
        json.dumps(runtime.unlock_template(protocol, owner="zi wang")),
        encoding="utf-8",
    )
    return path


def _valid_provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
    return (
        {"action": "answer", "answer": ""},
        {"input_tokens": 10, "output_tokens": 5},
        1,
        4.0,
        "provider-request-hash",
    )


def _dotenv(path: Path, provider: str = "OpenAI") -> Path:
    key = {
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
    }[provider]
    path.write_text(f"{key}=not-a-real-key\n", encoding="utf-8")
    return path


def test_execution_protocol_is_hash_bound_and_validate_is_zero_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    monkeypatch.setattr(runtime, "_head_commit", lambda: "offline-test-commit")
    monkeypatch.setattr(runtime, "_contract_dirty_paths", lambda: ())

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("validation must not read credentials or call a provider")

    monkeypatch.setattr(runtime.provider_runtime, "_load_dotenv", forbidden)
    for provider in runtime.PROVIDER_CALLS:
        monkeypatch.setitem(runtime.PROVIDER_CALLS, provider, forbidden)
    receipt = runtime.validate_execution_protocol(protocol)

    assert receipt["request_count_per_provider"] == 384
    assert receipt["input_byte_bound_total"] == 514818
    assert receipt["max_input_byte_bound_per_request"] == 1528
    assert receipt["max_output_tokens_per_request"] == 192
    assert receipt["total_hard_cap_usd"] == 8.5
    assert {row["provider"] for row in receipt["providers"]} == {
        "OpenAI",
        "Gemini",
        "DeepSeek",
    }
    assert receipt["credential_read"] is False
    assert receipt["provider_call_made"] is False


def test_unlock_binds_commit_protocol_models_owner_and_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    unlock_path = _unlock(protocol, tmp_path / "unlock.json", monkeypatch)

    assert runtime.validate_unlock(protocol, unlock_path)["owner"] == "zi wang"
    changed = json.loads(unlock_path.read_text(encoding="utf-8"))
    changed["total_hard_cap_usd"] = 8.51
    unlock_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact contract"):
        runtime.validate_unlock(protocol, unlock_path)

    monkeypatch.setattr(runtime, "_contract_dirty_paths", lambda: ("scripts/changed.py",))
    with pytest.raises(RuntimeError, match="contract-bearing paths are dirty"):
        runtime.validate_unlock(protocol, unlock_path)


def test_gemini_cost_usage_includes_thinking_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    binding = runtime._binding(protocol, "Gemini")

    def fake_http(**_kwargs: object) -> tuple[object, ...]:
        return (
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": json.dumps({"action": "answer", "answer": "complete"})}
                            ]
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 20,
                    "totalTokenCount": 34,
                },
            },
            1,
            2.0,
        )

    monkeypatch.setattr(runtime.provider_runtime, "_http_json", fake_http)
    response, usage, _attempts, _latency, _request_hash = runtime._gemini_exposure_request(
        binding,
        api_key="not-a-real-key",
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
        max_output_tokens=192,
        timeout_seconds=120,
        max_retries=0,
    )

    assert response["answer"] == "complete"
    assert usage == {"input_tokens": 10, "output_tokens": 24}


def test_missing_credential_does_not_create_frozen_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    unlock_path = _unlock(protocol, tmp_path / "unlock.json", monkeypatch)
    binding = runtime._binding(protocol, "OpenAI")
    calls = 0

    def provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return _valid_provider()

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", provider)
    runtime_dir = tmp_path / "runtime"
    with pytest.raises(FileNotFoundError, match="dotenv"):
        runtime.fixture_provider(
            protocol,
            binding=binding,
            unlock_path=unlock_path,
            dotenv=tmp_path / "missing.env",
            runtime_dir=runtime_dir,
        )

    assert calls == 0
    assert not runtime._fixture_failure_path(runtime_dir, binding).exists()


def test_checkpoint_resumes_after_interrupt_and_scores_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    protocol = replace(
        base,
        base=replace(base.base, bootstrap_replicates=20),
    )
    unlock_path = _unlock(protocol, tmp_path / "unlock.json", monkeypatch)
    binding = runtime._binding(protocol, "OpenAI")
    dotenv = _dotenv(tmp_path / ".env")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", _valid_provider)
    first_fixture = runtime.fixture_provider(
        protocol,
        binding=binding,
        unlock_path=unlock_path,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )
    repeated_fixture = runtime.fixture_provider(
        protocol,
        binding=binding,
        unlock_path=unlock_path,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )
    assert first_fixture == repeated_fixture

    calls_before_interrupt = 0

    def interrupting(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls_before_interrupt
        if calls_before_interrupt == 7:
            raise KeyboardInterrupt
        calls_before_interrupt += 1
        return _valid_provider()

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", interrupting)
    with pytest.raises(KeyboardInterrupt):
        runtime.execute_provider(
            protocol,
            binding=binding,
            unlock_path=unlock_path,
            dotenv=dotenv,
            runtime_dir=runtime_dir,
        )
    checkpoint = runtime._checkpoint_path(runtime_dir, binding)
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 7
    assert not runtime._execution_failure_path(runtime_dir, binding).exists()

    resumed_calls = 0

    def resumed(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal resumed_calls
        resumed_calls += 1
        return _valid_provider()

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", resumed)
    completion = runtime.execute_provider(
        protocol,
        binding=binding,
        unlock_path=unlock_path,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )
    assert completion["completed_requests"] == 384
    assert resumed_calls == 377

    def no_more_calls(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("completed execution and local scoring must make no calls")

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", no_more_calls)
    repeated = runtime.execute_provider(
        protocol,
        binding=binding,
        unlock_path=unlock_path,
        dotenv=tmp_path / "missing.env",
        runtime_dir=runtime_dir,
    )
    assert repeated == completion
    manifest = runtime.score_provider(
        protocol,
        binding=binding,
        unlock_path=unlock_path,
        runtime_dir=runtime_dir,
    )
    assert manifest["scored_pair_count"] == 192
    assert manifest["expected_request_count"] == 384
    assert manifest["provider_call_made_during_scoring"] is False


def test_invalid_response_freezes_and_prohibits_selective_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    unlock_path = _unlock(protocol, tmp_path / "unlock.json", monkeypatch)
    binding = runtime._binding(protocol, "OpenAI")
    dotenv = _dotenv(tmp_path / ".env")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", _valid_provider)
    runtime.fixture_provider(
        protocol,
        binding=binding,
        unlock_path=unlock_path,
        dotenv=dotenv,
        runtime_dir=runtime_dir,
    )
    calls = 0

    def invalid(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return (
            {"action": "unsupported", "answer": ""},
            {"input_tokens": 10, "output_tokens": 5},
            1,
            4.0,
            "provider-request-hash",
        )

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", invalid)
    with pytest.raises(ValueError, match="reader action"):
        runtime.execute_provider(
            protocol,
            binding=binding,
            unlock_path=unlock_path,
            dotenv=dotenv,
            runtime_dir=runtime_dir,
        )
    failure = json.loads(
        runtime._execution_failure_path(runtime_dir, binding).read_text(encoding="utf-8")
    )
    assert failure["automatic_rerun_allowed"] is False
    assert failure["selective_rerun_allowed"] is False
    assert failure["output_repair_attempted"] is False

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", _valid_provider)
    with pytest.raises(RuntimeError, match="selective rerun is prohibited"):
        runtime.execute_provider(
            protocol,
            binding=binding,
            unlock_path=unlock_path,
            dotenv=dotenv,
            runtime_dir=runtime_dir,
        )
    assert calls == 1


def test_conservative_cap_blocks_before_credential_or_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = runtime.load_execution_protocol(runtime.DEFAULT_PROTOCOL)
    unlock_path = _unlock(protocol, tmp_path / "unlock.json", monkeypatch)
    binding = replace(runtime._binding(protocol, "OpenAI"), hard_cap_usd=0.0)
    calls = 0

    def provider(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return _valid_provider()

    monkeypatch.setitem(runtime.PROVIDER_CALLS, "OpenAI", provider)
    with pytest.raises(RuntimeError, match="conservative cost"):
        runtime.fixture_provider(
            protocol,
            binding=binding,
            unlock_path=unlock_path,
            dotenv=tmp_path / "missing.env",
            runtime_dir=tmp_path / "runtime",
        )
    assert calls == 0
