"""Run a fixed-ranking top-k recall-risk-cost frontier."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.run_retrieval_experiment import load_cases, load_protocol
from verify_agent_memory.pareto import run_top_k_pareto
from verify_agent_memory.retrieval import RetrievalArm
from verify_agent_memory.serialization import setting_summary_to_mapping


def _mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--retrieval-protocol", type=Path, required=True)
    parser.add_argument("--pareto-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.cases)
    target_recall, _, all_configs = load_protocol(args.retrieval_protocol)
    protocol = _mapping(args.pareto_protocol)
    if protocol.get("schema_version") != 1:
        raise ValueError("pareto protocol schema_version must be 1")
    protocol_target = float(protocol.get("target_recall"))
    if not math.isclose(target_recall, protocol_target, abs_tol=1e-15):
        raise ValueError("pareto and retrieval target_recall values must match")
    fixed_arms = {
        RetrievalArm(str(value)) for value in _sequence(protocol.get("fixed_arms"), "fixed_arms")
    }
    configs = tuple(config for config in all_configs if config.arm in fixed_arms)
    if len(configs) != len(fixed_arms) or {config.arm for config in configs} != fixed_arms:
        raise ValueError("fixed_arms need exactly one setting each in the retrieval protocol")
    top_k_values = tuple(int(value) for value in _sequence(protocol.get("top_k"), "top_k"))
    points = run_top_k_pareto(
        cases,
        configs,
        top_k_values,
        target_recall=target_recall,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                {
                    "base_setting_id": point.base_setting_id,
                    "top_k": point.top_k,
                    **setting_summary_to_mapping(point.summary),
                    "mean_returned_count": point.mean_returned_count,
                    "mean_candidates_scored": point.mean_candidates_scored,
                    "mean_route_width": point.mean_route_width,
                    "mean_latency_ms": point.mean_latency_ms,
                    "latency_claim_eligible": False,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for point in points
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"rows": len(points), "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
