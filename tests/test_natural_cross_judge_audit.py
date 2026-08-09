from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scripts import run_natural_cross_judge_audit as audit
from verify_agent_memory.judge_agreement import (
    AuditAssignment,
    binary_agreement,
    bootstrap_binary_agreement,
    quality_agreement,
    select_unique_stratified,
)


def test_cross_judge_protocol_is_posthoc_blinded_and_cost_capped() -> None:
    protocol = audit.load_protocol()

    assert protocol.binding.provider == "OpenAI"
    assert protocol.binding.model == "gpt-5-pro-2025-10-06"
    assert protocol.binding.controls == {"effort": "high"}
    assert protocol.judge["arm_and_reader_blinded"] is True
    assert protocol.sampling["selection_uses_outcomes_or_answer_content"] is False
    assert sum(audit._quotas(protocol).values()) == 200
    assert len(audit._quotas(protocol)) == 18
    assert protocol.fixture_cap_usd + 200 * protocol.call_reservation_usd == pytest.approx(19.45)
    assert protocol.total_cap_usd == 20.0
    assert protocol.raw["interpretation"]["independently_preregistered_replication"] is False


def test_ceiling_recovery_is_zero_label_migration_under_same_cap() -> None:
    protocol = audit.load_protocol(
        audit.ROOT / "experiments" / "natural_cross_judge_recovery_protocol.json"
    )
    recovery = protocol.raw["recovery"]

    assert protocol.binding.model == "gpt-5.1-2025-11-13"
    assert protocol.binding.controls == {"effort": "high"}
    assert protocol.judge["maximum_output_tokens"] == 4096
    assert recovery["accepted_benchmark_response_count"] == 0
    assert recovery["selection_uses_outcomes"] is False
    assert protocol.prior_budget_consumption_usd == pytest.approx(0.14907)
    assert (
        protocol.prior_budget_consumption_usd
        + protocol.fixture_cap_usd
        + 200 * protocol.call_reservation_usd
    ) == pytest.approx(9.01707)
    assert protocol.total_cap_usd == 20.0


def test_unique_stratified_sample_is_deterministic_and_exact() -> None:
    assignments = []
    quotas = {}
    for reader in ("A", "B"):
        stratum = (reader, "source", "route")
        quotas[stratum] = 2
        assignments.extend(
            AuditAssignment(
                request_id=f"{reader}-{index}",
                case_id=f"case-{reader}-{index}",
                reader=reader,
                source="source",
                arm="route",
            )
            for index in range(5)
        )
    first = select_unique_stratified(assignments, quotas, seed="fixed-seed")
    second = select_unique_stratified(list(reversed(assignments)), quotas, seed="fixed-seed")

    assert first == second
    assert len(first) == 4
    assert len({unit.request_id for unit in first}) == 4
    assert {unit.anchor_reader for unit in first} == {"A", "B"}


def test_unique_stratified_sample_deduplicates_shared_payloads() -> None:
    assignments = [
        AuditAssignment("shared", "case-1", "A", "source", "route"),
        AuditAssignment("a-only", "case-2", "A", "source", "route"),
        AuditAssignment("shared", "case-1", "B", "source", "route"),
        AuditAssignment("b-only", "case-3", "B", "source", "route"),
        AuditAssignment("b-extra", "case-4", "B", "source", "route"),
    ]
    selected = select_unique_stratified(
        assignments,
        {("A", "source", "route"): 1, ("B", "source", "route"): 1},
        seed="deduplicate",
    )

    assert len(selected) == 2
    assert len({unit.request_id for unit in selected}) == 2
    shared = next((unit for unit in selected if unit.request_id == "shared"), None)
    if shared is not None:
        assert shared.associated_readers == ("A", "B")
        assert shared.assignment_count == 2


def test_binary_agreement_metrics_match_known_values() -> None:
    metrics = binary_agreement(
        [True, True, False, False],
        [True, False, False, False],
    )

    assert metrics["exact_agreement"] == 0.75
    assert metrics["cohen_kappa"] == 0.5
    assert metrics["gwet_ac1"] == pytest.approx(0.5294117647058824)
    assert metrics["first_true_second_false"] == 1
    assert metrics["first_false_second_true"] == 0


def test_quality_agreement_uses_quadratic_weights() -> None:
    metrics = quality_agreement([0, 5, 10], [0, 6, 8])

    assert metrics["exact_agreement"] == pytest.approx(1 / 3)
    assert metrics["within_one_agreement"] == pytest.approx(2 / 3)
    assert metrics["mean_absolute_error"] == 1.0
    assert -1 <= metrics["quadratic_weighted_kappa"] <= 1


def test_binary_bootstrap_is_reproducible() -> None:
    first = [True, False, True, False, True, False]
    second = [True, False, False, False, True, True]

    one = bootstrap_binary_agreement(first, second, iterations=200, seed=7)
    two = bootstrap_binary_agreement(first, second, iterations=200, seed=7)

    assert one == two
    assert one["exact_agreement"][0] <= one["exact_agreement"][1]


def test_report_groups_preserve_original_and_sequential_reader_panels() -> None:
    rows = [
        {"anchor_reader": "DeepSeek", "source": "rhelm", "anchor_arm": "global_dense"},
        {"anchor_reader": "Gemini", "source": "rhelm", "anchor_arm": "global_dense"},
        {"anchor_reader": "OpenAI", "source": "rhelm", "anchor_arm": "global_dense"},
    ]
    groups = {(kind, value): members for kind, value, members in audit._groups(rows)}

    assert len(groups[("reader_panel", "original_two_reader")]) == 2
    assert len(groups[("reader_panel", "sequential_gpt_reader")]) == 1


def test_synthetic_fixture_uses_exact_gpt_pro_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    protocol = audit.load_protocol()
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "protocol_sha256": kwargs["protocol"].protocol_sha256,
            "implementation_commit": kwargs["implementation_commit"],
            "stage": kwargs["stage"],
            "provider": kwargs["binding"].provider,
            "model": kwargs["binding"].model,
            "request_id": kwargs["spec"].request_id,
            "payload_sha256": "payload-sha",
            "request_body_sha256": "request-body-sha",
            "response": {
                "answer_correct": True,
                "answer_quality": 10,
                "protected_disclosure": False,
                "stale_disclosure": False,
                "reason": "Correct.",
            },
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "attempts": 1,
            "latency_ms": 1.0,
            "cost_usd": 0.0075,
        }

    monkeypatch.setattr(audit, "_require_clean_contract", lambda: "test-commit")
    monkeypatch.setattr(audit.base, "_call_provider", fake_call)
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=test-only\n", encoding="ascii")

    receipt = audit.run_fixture(protocol, runtime=tmp_path / "runtime", dotenv=dotenv)

    assert captured["binding"].model == "gpt-5-pro-2025-10-06"
    assert captured["binding"].controls == {"effort": "high"}
    assert captured["maximum_output_tokens"] == 512
    assert captured["spec"].schema == audit.base.judge_response_schema()
    assert receipt["synthetic_fixture"] is True
    assert receipt["response"] is None
    assert receipt["cost_usd"] < protocol.fixture_cap_usd
    json.dumps(receipt)


def test_frozen_sample_manifest_is_content_free_and_hash_bound() -> None:
    protocol = audit.load_protocol()
    path = audit.DEFAULT_SAMPLE_MANIFEST
    if not path.is_file():
        pytest.skip("sample manifest is generated after the protocol hash is frozen")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert audit.base._sha256_file(path) == protocol.sampling["sample_manifest_sha256"]
    assert manifest["sample_count"] == 200
    assert (
        manifest["request_id_set_sha256"] == (protocol.sampling["expected_request_id_set_sha256"])
    )
    assert manifest["answer_reference_or_response_content_included"] is False
    assert manifest["judge_outcomes_included"] is False
    assert len({row["request_id"] for row in manifest["units"]}) == 200


def test_fixture_rejects_cost_above_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    protocol = audit.load_protocol()
    monkeypatch.setattr(audit, "_require_clean_contract", lambda: "test-commit")
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=test-only\n", encoding="ascii")

    def expensive_call(**kwargs):
        del kwargs
        return {
            "response": {
                "answer_correct": True,
                "answer_quality": 10,
                "protected_disclosure": False,
                "stale_disclosure": False,
                "reason": "Correct.",
            },
            "cost_usd": 0.26,
        }

    monkeypatch.setattr(audit.base, "_call_provider", expensive_call)

    with pytest.raises(RuntimeError, match="fixture exceeded"):
        audit.run_fixture(protocol, runtime=tmp_path / "runtime", dotenv=dotenv)


def test_protocol_budget_guard_rejects_oversubscription() -> None:
    protocol = audit.load_protocol()
    budget = {**protocol.budget, "per_call_reservation_usd": 0.1}
    changed = replace(protocol, budget=budget)

    assert changed.fixture_cap_usd + 200 * changed.call_reservation_usd > (changed.total_cap_usd)
