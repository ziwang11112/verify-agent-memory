from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_claude_opus5_exposure_replication as replication


def test_protocol_is_opus_only_and_zero_call(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = replication.load_protocol(replication.DEFAULT_PROTOCOL)
    monkeypatch.setattr(replication.exposure_runtime, "_head_commit", lambda: "offline-test-commit")
    monkeypatch.setattr(replication.exposure_runtime, "_contract_dirty_paths", lambda: ())

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("validation must not read credentials or call a provider")

    monkeypatch.setattr(replication.provider_runtime, "_load_dotenv", forbidden)
    monkeypatch.setitem(replication.exposure_runtime.PROVIDER_CALLS, "Anthropic", forbidden)
    receipt = replication.validate_protocol(protocol)

    assert receipt["request_count_per_provider"] == 384
    assert receipt["max_output_tokens_per_request"] == 192
    assert receipt["total_hard_cap_usd"] == 5.0
    assert receipt["providers"] == [
        {"provider": "Anthropic", "model": "claude-opus-5", "hard_cap_usd": 5.0}
    ]
    assert receipt["thinking"] == "disabled"
    assert receipt["effort"] == "medium"
    assert receipt["changes_existing_results"] is False
    assert receipt["model_pooling"] is False
    assert receipt["credential_read"] is False
    assert receipt["provider_call_made"] is False


def test_opus_adapter_disables_thinking_and_uses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = replication.load_protocol(replication.DEFAULT_PROTOCOL)
    binding = replication.exposure_runtime._binding(protocol, "Anthropic")
    captured: dict[str, object] = {}

    def fake_http(**kwargs: object) -> tuple[object, ...]:
        captured.update(kwargs)
        return (
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"action": "answer", "answer": "complete"}),
                    }
                ],
                "usage": {"input_tokens": 21, "output_tokens": 13},
            },
            1,
            3.5,
        )

    monkeypatch.setattr(replication.provider_runtime, "_http_json", fake_http)
    response, usage, attempts, latency, _request_hash = replication._anthropic_opus5_request(
        binding,
        api_key="not-a-real-key",
        system_prompt="system",
        user_prompt="user",
        schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["action", "answer"],
            "additionalProperties": False,
        },
        max_output_tokens=192,
        timeout_seconds=120,
        max_retries=0,
    )

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "claude-opus-5"
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in body
    assert response == {"action": "answer", "answer": "complete"}
    assert usage == {"input_tokens": 21, "output_tokens": 13}
    assert attempts == 1
    assert latency == 3.5


def test_unlock_binds_clean_commit_model_protocol_and_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = replication.load_protocol(replication.DEFAULT_PROTOCOL)
    monkeypatch.setattr(replication.exposure_runtime, "_head_commit", lambda: "a" * 40)
    monkeypatch.setattr(replication.exposure_runtime, "_contract_dirty_paths", lambda: ())
    unlock = replication.exposure_runtime.unlock_template(protocol, owner="zi wang")
    unlock_path = tmp_path / "unlock.json"
    unlock_path.write_text(json.dumps(unlock), encoding="utf-8")

    checked = replication.exposure_runtime.validate_unlock(protocol, unlock_path)
    assert checked["models"] == {"Anthropic": "claude-opus-5"}
    assert checked["total_hard_cap_usd"] == 5.0

    changed = dict(unlock)
    changed["total_hard_cap_usd"] = 5.01
    unlock_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact contract"):
        replication.exposure_runtime.validate_unlock(protocol, unlock_path)


def test_paid_command_is_blocked_before_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = replication.load_protocol(replication.DEFAULT_PROTOCOL)
    monkeypatch.setattr(replication.exposure_runtime, "_contract_dirty_paths", lambda: ())
    with pytest.raises((FileNotFoundError, RuntimeError)):
        replication._run_command(
            protocol,
            command="preflight",
            unlock_path=tmp_path / "missing-unlock.json",
            dotenv=tmp_path / ".env",
            runtime_dir=tmp_path / "runtime",
        )
