from __future__ import annotations

from scripts import publish_natural_case_audit as audit


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "case_token": audit._token(audit.CASE_TOKEN_DOMAIN, "case-1"),
        "group_token": audit._token(audit.GROUP_TOKEN_DOMAIN, "rhelm|group-1"),
        "source": "rhelm",
        "reader_provider": "Reader",
        "reader_model": "reader-model",
        "arm": "global_dense",
        "case_order": 0,
        "anchor_total": 2,
        **{field: 0.0 for field in audit.SCORE_FIELDS},
        "non_answer": 0,
        "reader_record_sha256": "a" * 64,
        "judge_record_sha256": "b" * 64,
    }
    row.update(overrides)
    return row


def test_sanitized_row_excludes_raw_identifiers_and_text() -> None:
    raw = {
        "case_id": "private-case",
        "group_id": "private-group",
        "source": "rhelm",
        "reader_provider": "Reader",
        "reader_model": "reader-model",
        "arm": "global_dense",
        "over_refusal": 1,
        **{field: 0.0 for field in audit.SCORE_FIELDS},
    }
    sanitized = audit._sanitize_row(
        raw,
        case_order=0,
        anchor_total=2,
        reader_record_sha256="a" * 64,
        judge_record_sha256="b" * 64,
    )

    assert sanitized["case_token"] != raw["case_id"]
    assert sanitized["group_token"] != raw["group_id"]
    assert sanitized["non_answer"] == 1
    assert "case_id" not in sanitized
    assert "group_id" not in sanitized
    assert "over_refusal" not in sanitized


def test_source_delta_pairs_cases_and_resamples_groups() -> None:
    rows = []
    for index, (baseline, namespace) in enumerate(((0.0, 1.0), (0.5, 1.0)), start=1):
        for arm, value in (("global_dense", baseline), ("namespace_dense", namespace)):
            rows.append(
                _row(
                    case_token=audit._token(audit.CASE_TOKEN_DOMAIN, f"case-{index}"),
                    group_token=audit._token(audit.GROUP_TOKEN_DOMAIN, f"rhelm|group-{index}"),
                    arm=arm,
                    answer_correct=value,
                )
            )

    point, lower, upper, pair_count, group_count = audit._source_delta(
        rows,
        provider="Reader",
        source="rhelm",
        arm="namespace_dense",
        reference="global_dense",
        metric="answer_correct",
    )

    assert point == 0.75
    assert lower == 0.5
    assert upper == 1.0
    assert pair_count == 2
    assert group_count == 2


def test_case_weighted_sensitivity_weights_cases_not_sources() -> None:
    rows = []
    cases = (
        ("rhelm", "rhelm-group", 1.0),
        ("memops", "memops-group-1", 0.0),
        ("memops", "memops-group-2", 0.0),
        ("memops", "memops-group-3", 0.0),
    )
    for index, (source, group, namespace_value) in enumerate(cases, start=1):
        for arm, value in (("global_dense", 0.0), ("namespace_dense", namespace_value)):
            rows.append(
                _row(
                    case_token=audit._token(audit.CASE_TOKEN_DOMAIN, f"case-{index}"),
                    group_token=audit._token(audit.GROUP_TOKEN_DOMAIN, f"{source}|{group}"),
                    source=source,
                    arm=arm,
                    answer_correct=value,
                )
            )

    point, lower, upper, pair_count, group_count = audit._case_weighted_delta(
        rows,
        provider="Reader",
        arm="namespace_dense",
        reference="global_dense",
        metric="answer_correct",
    )

    assert point == 0.25
    assert lower <= point <= upper
    assert pair_count == 4
    assert group_count == 4


def test_public_package_verifies_without_private_runtime() -> None:
    result = audit.verify()

    assert result["status"] == "verified"
    assert result["case_score_rows"] == 19_799
    assert result["provider_calls_made"] == 0
    assert result["raw_content_read"] is False
