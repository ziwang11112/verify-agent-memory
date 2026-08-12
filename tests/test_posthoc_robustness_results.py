from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_support_control_bundle_is_hash_bound_and_content_free() -> None:
    result_dir = ROOT / "results" / "support_controls"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["analysis"] == "natural-support-controls-v1"
    assert manifest["query_count"] == 3767
    assert manifest["group_count"] == 87
    assert manifest["contains_query_memory_or_group_ids"] is False
    assert manifest["contains_raw_text_embeddings_prompts_or_responses"] is False
    assert manifest["evaluation_retuning"] is False
    assert manifest["provider_calls"] == manifest["reader_calls"] == 0
    assert manifest["judge_calls"] == manifest["paid_calls"] == 0

    forbidden = {"query_id", "memory_id", "group_id", "query_text", "response", "embedding"}
    for name, receipt in manifest["published_files"].items():
        path = result_dir / name
        assert _sha256(path) == receipt["sha256"]
        if path.suffix == ".csv":
            rows = _csv_rows(path)
            assert len(rows) == receipt["row_count"]
            assert not forbidden.intersection(rows[0])


def test_operating_curve_bundle_is_hash_bound_and_content_free() -> None:
    result_dir = ROOT / "results" / "posthoc_robustness"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["analysis"] == "posthoc-operating-curves-and-cluster-robustness-v1"
    assert manifest["contains_case_candidate_scenario_or_group_ids"] is False
    assert manifest["contains_raw_text_prompts_or_responses"] is False
    assert manifest["evaluation_retuning"] is False
    assert manifest["model_pooling"] is False
    assert manifest["provider_calls"] == manifest["reader_calls"] == 0
    assert manifest["judge_calls"] == manifest["paid_calls"] == 0

    forbidden = {"case_id", "candidate_id", "scenario_id", "group_id", "prompt", "response"}
    for name, receipt in manifest["outputs"].items():
        path = result_dir / name
        assert _sha256(path) == receipt["sha256"]
        if path.suffix == ".csv":
            rows = _csv_rows(path)
            assert len(rows) == receipt["rows"]
            assert not forbidden.intersection(rows[0])
