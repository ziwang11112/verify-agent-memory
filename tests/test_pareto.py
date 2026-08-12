from __future__ import annotations

import json
from pathlib import Path

from scripts.run_retrieval_experiment import load_cases
from scripts.run_top_k_pareto import main
from verify_agent_memory.pareto import run_top_k_pareto
from verify_agent_memory.retrieval import RetrievalArm, RetrievalConfig

ROOT = Path(__file__).resolve().parents[1]


def test_top_k_pareto_uses_nested_ranking_prefixes() -> None:
    cases = load_cases(ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl")
    config = RetrievalConfig("dense", RetrievalArm.GLOBAL_DENSE, top_k=100)

    points = run_top_k_pareto(cases, (config,), (1, 2), target_recall=0.8)

    assert [point.top_k for point in points] == [1, 2]
    assert points[0].mean_returned_count == 1.0
    assert points[1].mean_returned_count == 2.0
    assert points[0].mean_candidates_scored == points[1].mean_candidates_scored
    assert points[1].summary.feasible_rate >= points[0].summary.feasible_rate


def test_top_k_pareto_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "pareto.jsonl"
    protocol = tmp_path / "pareto.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_recall": 0.8,
                "top_k": [1, 2],
                "fixed_arms": ["global_dense", "namespace_dense"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--cases",
                str(ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl"),
                "--retrieval-protocol",
                str(ROOT / "experiments" / "frozen_natural_protocol.json"),
                "--pareto-protocol",
                str(protocol),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["top_k"] for row in rows} == {1, 2}
    assert all(row["latency_claim_eligible"] is False for row in rows)
