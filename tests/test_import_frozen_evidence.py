from __future__ import annotations

import pytest

from scripts.import_frozen_evidence import (
    NATURAL_EXECUTION_COMMIT,
    _source_macro,
    _validate_natural_execution_manifest,
)
from verify_agent_memory.provenance import sha256_bytes


def test_source_macro_uses_only_base_source_rows() -> None:
    rows = [
        {"arm": "flat", "source": "rhelm", "recall": "0.6"},
        {"arm": "flat", "source": "memops", "recall": "0.8"},
        {"arm": "flat", "source": "source_macro", "recall": "0.7"},
        {"arm": "flat", "source": "pooled_queries_sensitivity", "recall": "0.75"},
    ]
    assert _source_macro(rows, arm="flat", metric="recall") == pytest.approx(0.7)


def test_source_macro_requires_one_row_per_base_source() -> None:
    rows = [
        {"arm": "flat", "source": "rhelm", "recall": "0.6"},
        {"arm": "flat", "source": "rhelm", "recall": "0.7"},
        {"arm": "flat", "source": "memops", "recall": "0.8"},
    ]
    with pytest.raises(ValueError, match="exactly the two expected sources"):
        _source_macro(rows, arm="flat", metric="recall")


def natural_manifest(main: bytes, paired: bytes) -> dict[str, object]:
    arms = {
        "flat_bm25",
        "flat_bm25_dense_rrf",
        "flat_dense",
        "namespace_current_only",
        "namespace_dense",
        "namespace_source_intent_lifecycle",
        "ncr_a5",
        "ncr_threshold",
        "recency_dense",
    }
    return {
        "commit": NATURAL_EXECUTION_COMMIT,
        "query_rows": 3,
        "route_rows": 3 * len(arms),
        "score_rows": 3 * len(arms),
        "provider_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
        "paid_calls": 0,
        "selected_settings": {arm: "setting" for arm in arms},
        "artifacts": {
            "main_table.csv": sha256_bytes(main),
            "paired_deltas.csv": sha256_bytes(paired),
        },
    }


def test_natural_manifest_proves_complete_method_query_coverage() -> None:
    main = b"main"
    paired = b"paired"
    assert (
        _validate_natural_execution_manifest(
            natural_manifest(main, paired),
            main_content=main,
            paired_content=paired,
        )
        == 3
    )


def test_natural_manifest_rejects_incomplete_coverage() -> None:
    main = b"main"
    paired = b"paired"
    manifest = natural_manifest(main, paired)
    manifest["score_rows"] = 26
    with pytest.raises(ValueError, match="cover every method and query"):
        _validate_natural_execution_manifest(
            manifest,
            main_content=main,
            paired_content=paired,
        )


def test_natural_manifest_rejects_artifact_drift() -> None:
    main = b"main"
    paired = b"paired"
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_natural_execution_manifest(
            natural_manifest(main, paired),
            main_content=b"changed",
            paired_content=paired,
        )
