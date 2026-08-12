from __future__ import annotations

from pathlib import Path

from scripts.publish_inferred_admissibility_results import verify

ROOT = Path(__file__).resolve().parents[1]


def test_published_inferred_admissibility_results_are_hash_bound_and_content_free() -> None:
    verify(ROOT / "results" / "inferred_admissibility")
