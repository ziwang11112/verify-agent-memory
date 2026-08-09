from __future__ import annotations

from pathlib import Path

from scripts.import_natural_end_to_end_evidence import normalized_rows

ROOT = Path(__file__).resolve().parents[1]


def rows() -> tuple[dict[str, object], ...]:
    return normalized_rows(
        ROOT / "results" / "natural_end_to_end_two_reader_judged",
        ROOT / "results" / "natural_end_to_end_gpt_luna_judged",
    )


def test_natural_end_to_end_import_keeps_readers_separate() -> None:
    imported = rows()
    assert len(imported) == 34
    assert {row["source"] for row in imported} == {
        "deepseek-v4-pro",
        "gemini-3.6-flash",
        "gpt-5.6-luna",
    }
    assert all(row["claim_id"] == "C13" for row in imported)
    assert all("reader_estimates_not_pooled" in str(row["notes"]) for row in imported)


def test_gpt_rows_preserve_sequential_replication_boundary() -> None:
    gpt_rows = [row for row in rows() if row["source"] == "gpt-5.6-luna"]
    assert len(gpt_rows) == 10
    assert all("sequential_reader_replication" in str(row["notes"]) for row in gpt_rows)
    assert not any("namespace_text_verifier" in str(row["contrast"]) for row in gpt_rows)


def test_disclosure_intervals_are_preserved_as_null_results() -> None:
    disclosure_rows = [
        row
        for row in rows()
        if row["contrast"] == "namespace_dense_minus_global_dense"
        and row["metric"] in {"protected_disclosure", "stale_disclosure"}
    ]
    assert len(disclosure_rows) == 6
    assert all(float(row["ci95_lower"]) <= 0 <= float(row["ci95_upper"]) for row in disclosure_rows)
