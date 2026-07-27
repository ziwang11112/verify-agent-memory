from __future__ import annotations

from pathlib import Path

from scripts.build_paper_artifacts import _evidence_rows, _load_rows
from scripts.verify_paper import validate_paper

ROOT = Path(__file__).resolve().parents[1]


def test_paper_evidence_selectors_cover_primary_claims() -> None:
    selected = _evidence_rows(_load_rows(ROOT))
    assert selected["reader_a"]["claim_id"] == "C2"
    assert selected["g1_leakage"]["claim_id"] == "C3"
    assert selected["smoke_global_recall"]["claim_id"] == "C4"
    assert selected["namespace_recall_delta"]["claim_id"] == "C5"
    assert selected["threshold_recall_delta"]["claim_id"] == "C6"
    assert selected["lifecycle_contamination_delta"]["claim_id"] == "C7"
    assert selected["prohibited_alpha"]["claim_id"] == "C8"


def test_paper_package_passes() -> None:
    assert validate_paper(ROOT) == []
