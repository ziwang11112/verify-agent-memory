"""Run a zero-provider metadata-corruption curve over frozen retrieval settings."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.run_retrieval_experiment import load_cases, load_protocol
from verify_agent_memory.experiment import SelectionRisk
from verify_agent_memory.retrieval import RetrievalArm
from verify_agent_memory.robustness import (
    CorruptionChannel,
    MetadataCorruption,
    break_even_interval,
    run_corruption_curve,
)
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


def _corruptions(protocol: Mapping[str, object]) -> tuple[MetadataCorruption, ...]:
    channels = [
        CorruptionChannel(str(value)) for value in _sequence(protocol.get("channels"), "channels")
    ]
    rates = [float(value) for value in _sequence(protocol.get("rates"), "rates")]
    seeds = [int(value) for value in _sequence(protocol.get("seeds"), "seeds")]
    return tuple(
        MetadataCorruption(
            corruption_id=f"{channel.value}:rate={rate:.6f}:seed={seed}",
            channel=channel,
            rate=rate,
            seed=seed,
        )
        for channel in channels
        for seed in seeds
        for rate in rates
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--retrieval-protocol", type=Path, required=True)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--break-even-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.cases)
    target_recall, _, all_configs = load_protocol(args.retrieval_protocol)
    protocol = _mapping(args.robustness_protocol)
    if protocol.get("schema_version") != 1:
        raise ValueError("robustness protocol schema_version must be 1")
    try:
        robustness_target_recall = float(protocol.get("target_recall"))
    except (TypeError, ValueError) as error:
        raise TypeError("robustness target_recall must be numeric") from error
    if not math.isclose(target_recall, robustness_target_recall, abs_tol=1e-15):
        raise ValueError("robustness and retrieval target_recall values must match")
    risk_target = SelectionRisk(str(protocol.get("selection_risk")))
    if risk_target is not SelectionRisk.ADMISSIBILITY_UPPER_BOUND:
        raise ValueError("metadata break-even requires admissibility_upper_bound risk")
    fixed_arms = {
        RetrievalArm(str(value)) for value in _sequence(protocol.get("fixed_arms"), "fixed_arms")
    }
    configs = tuple(config for config in all_configs if config.arm in fixed_arms)
    if len(configs) != len(fixed_arms) or {config.arm for config in configs} != fixed_arms:
        raise ValueError("fixed_arms need exactly one setting each in the retrieval protocol")
    corruptions = _corruptions(protocol)
    points = run_corruption_curve(
        cases,
        configs,
        corruptions,
        target_recall=target_recall,
    )
    _write_jsonl(
        args.output,
        [
            {
                "corruption_id": point.corruption_id,
                "channel": point.channel.value,
                "rate": point.rate,
                "seed": point.seed,
                **setting_summary_to_mapping(point.setting),
            }
            for point in points
        ],
    )

    namespace_channels = tuple(
        sorted(
            {point.channel for point in points if point.channel.value.startswith("namespace_")},
            key=lambda channel: channel.value,
        )
    )
    seeds = sorted({point.seed for point in points})
    intervals = [
        break_even_interval(
            [point for point in points if point.seed == seed],
            channel=channel,
        )
        for channel in namespace_channels
        for seed in seeds
    ]
    _write_jsonl(
        args.break_even_output,
        [
            {
                "channel": interval.channel.value,
                "treatment_arm": interval.treatment_arm.value,
                "reference_arm": interval.reference_arm.value,
                "seed": seed,
                "last_dominating_rate": interval.last_dominating_rate,
                "first_non_dominating_rate": interval.first_non_dominating_rate,
            }
            for interval, seed in zip(
                intervals,
                [seed for _ in namespace_channels for seed in seeds],
                strict=True,
            )
        ],
    )
    print(
        json.dumps(
            {
                "break_even_rows": len(intervals),
                "corruption_settings": len(corruptions),
                "robustness_rows": len(points),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
