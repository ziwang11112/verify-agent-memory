"""Publish a content-free posthoc taxonomy of counterfactual verifier errors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import run_counterfactual_admissibility_experiment as runtime
from verify_agent_memory.counterfactual_admissibility import (
    CONDITIONS,
    CounterfactualPair,
    CounterfactualPrediction,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "counterfactual_admissibility"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"CSV rows for {path} have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _write_text(
        path,
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def candidate_observations(
    pairs: Sequence[CounterfactualPair],
    predictions: Mapping[str, CounterfactualPrediction],
    *,
    model: str,
) -> tuple[dict[str, object], ...]:
    """Return private in-memory observations; callers publish aggregates only."""
    rows = []
    for pair in pairs:
        for condition in CONDITIONS:
            prediction = predictions.get(pair.case_id(condition))
            if prediction is None:
                raise ValueError(f"missing prediction for {pair.case_id(condition)!r}")
            predicted_by_key = {
                candidate.candidate_key: candidate for candidate in prediction.candidates
            }
            expected_keys = tuple(candidate.candidate_key for candidate in pair.candidates)
            if tuple(predicted_by_key) != expected_keys:
                raise ValueError("prediction candidate order differs from pair")
            for candidate in pair.candidates:
                candidate_prediction = predicted_by_key[candidate.candidate_key]
                probabilities = candidate_prediction.probabilities
                gold = "admissible" if candidate.gold(condition) else "inadmissible"
                predicted = probabilities.argmax()
                rows.append(
                    {
                        "model": model,
                        "pair_id": pair.pair_id,
                        "scenario_id": pair.scenario_id,
                        "axis": pair.axis,
                        "role": candidate.role,
                        "condition": condition,
                        "gold": gold,
                        "predicted": predicted,
                        "correct": int(predicted == gold),
                        "p_admissible": probabilities.probability("admissible"),
                        "p_inadmissible": probabilities.probability("inadmissible"),
                        "p_unknown": probabilities.probability("unknown"),
                    }
                )
    return tuple(rows)


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_judgments(
    observations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Aggregate false admission, false denial, and unknown rates by axis and role."""
    if not observations:
        raise ValueError("judgment taxonomy requires observations")
    model = _string(observations[0]["model"], "model")
    if any(row["model"] != model for row in observations):
        raise ValueError("judgment taxonomy requires one model at a time")
    axes = ("all", *sorted({_string(row["axis"], "axis") for row in observations}))
    roles = ("focal", "stable_admissible", "stable_inadmissible")
    conditions = ("all", *CONDITIONS)
    results = []
    for axis in axes:
        axis_rows = [row for row in observations if axis == "all" or row["axis"] == axis]
        for role in roles:
            role_rows = [row for row in axis_rows if row["role"] == role]
            for condition in conditions:
                rows = [
                    row for row in role_rows if condition == "all" or row["condition"] == condition
                ]
                if not rows:
                    continue
                admissible_gold = [row for row in rows if row["gold"] == "admissible"]
                inadmissible_gold = [row for row in rows if row["gold"] == "inadmissible"]
                false_denies = sum(
                    row["gold"] == "admissible" and row["predicted"] == "inadmissible"
                    for row in rows
                )
                false_admits = sum(
                    row["gold"] == "inadmissible" and row["predicted"] == "admissible"
                    for row in rows
                )
                results.append(
                    {
                        "model": model,
                        "axis": axis,
                        "role": role,
                        "condition": condition,
                        "judgment_count": len(rows),
                        "gold_admissible_count": len(admissible_gold),
                        "gold_inadmissible_count": len(inadmissible_gold),
                        "accuracy": _mean(rows, "correct"),
                        "false_deny_rate": (
                            false_denies / len(admissible_gold) if admissible_gold else None
                        ),
                        "false_admit_rate": (
                            false_admits / len(inadmissible_gold) if inadmissible_gold else None
                        ),
                        "unknown_argmax_rate": sum(row["predicted"] == "unknown" for row in rows)
                        / len(rows),
                        "mean_p_admissible": _mean(rows, "p_admissible"),
                        "mean_p_inadmissible": _mean(rows, "p_inadmissible"),
                        "mean_p_unknown": _mean(rows, "p_unknown"),
                    }
                )
    return tuple(results)


def stable_pair_observations(
    observations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Collapse the two conditions for every stable control without publishing IDs."""
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in observations:
        if row["role"] == "focal":
            continue
        key = _string(row["pair_id"], "pair_id"), _string(row["role"], "role")
        grouped[key][_string(row["condition"], "condition")] = row
    results = []
    for condition_rows in grouped.values():
        if set(condition_rows) != set(CONDITIONS):
            raise ValueError("stable control is missing one counterfactual condition")
        allow = condition_rows["allow"]
        block = condition_rows["block"]
        if (
            allow["model"] != block["model"]
            or allow["axis"] != block["axis"]
            or allow["role"] != block["role"]
        ):
            raise ValueError("stable control identity drifted across conditions")
        allow_prediction = _string(allow["predicted"], "allow prediction")
        block_prediction = _string(block["predicted"], "block prediction")
        results.append(
            {
                "model": allow["model"],
                "axis": allow["axis"],
                "role": allow["role"],
                "allow_prediction": allow_prediction,
                "block_prediction": block_prediction,
                "allow_correct": int(allow["correct"]),
                "block_correct": int(block["correct"]),
                "both_correct": int(allow["correct"] and block["correct"]),
                "overflip": int(allow_prediction != block_prediction),
                "same_wrong": int(
                    not allow["correct"]
                    and not block["correct"]
                    and allow_prediction == block_prediction
                ),
            }
        )
    return tuple(results)


def aggregate_stable_pairs(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Aggregate stable-control selectivity and directional transition counts."""
    if not rows:
        raise ValueError("stable-control taxonomy requires rows")
    model = _string(rows[0]["model"], "model")
    axes = ("all", *sorted({_string(row["axis"], "axis") for row in rows}))
    roles = ("all_stable", "stable_admissible", "stable_inadmissible")
    results = []
    for axis in axes:
        axis_rows = [row for row in rows if axis == "all" or row["axis"] == axis]
        for role in roles:
            selected = [row for row in axis_rows if role == "all_stable" or row["role"] == role]
            results.append(
                {
                    "model": model,
                    "axis": axis,
                    "role": role,
                    "stable_candidate_pair_count": len(selected),
                    "allow_accuracy": _mean(selected, "allow_correct"),
                    "block_accuracy": _mean(selected, "block_correct"),
                    "both_correct_rate": _mean(selected, "both_correct"),
                    "overflip_rate": _mean(selected, "overflip"),
                    "same_wrong_rate": _mean(selected, "same_wrong"),
                }
            )
    return tuple(results)


def aggregate_transitions(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Count stable-control prediction transitions globally and by axis."""
    if not rows:
        raise ValueError("transition taxonomy requires rows")
    model = _string(rows[0]["model"], "model")
    axes = ("all", *sorted({_string(row["axis"], "axis") for row in rows}))
    roles = ("all_stable", "stable_admissible", "stable_inadmissible")
    results = []
    for axis in axes:
        axis_rows = [row for row in rows if axis == "all" or row["axis"] == axis]
        for role in roles:
            selected = [row for row in axis_rows if role == "all_stable" or row["role"] == role]
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for row in selected:
                counts[
                    (
                        _string(row["allow_prediction"], "allow_prediction"),
                        _string(row["block_prediction"], "block_prediction"),
                    )
                ] += 1
            for (allow_prediction, block_prediction), count in sorted(counts.items()):
                results.append(
                    {
                        "model": model,
                        "axis": axis,
                        "role": role,
                        "allow_prediction": allow_prediction,
                        "block_prediction": block_prediction,
                        "transition_count": count,
                        "transition_rate": count / len(selected),
                    }
                )
    return tuple(results)


def _overall_taxonomy(
    rows: Sequence[Mapping[str, object]],
    *,
    model: str,
    role: str,
) -> Mapping[str, object]:
    matches = [
        row
        for row in rows
        if row["model"] == model
        and row["axis"] == "all"
        and row["role"] == role
        and row["condition"] == "all"
    ]
    if len(matches) != 1:
        raise ValueError("overall taxonomy row is missing or duplicated")
    return matches[0]


def _overall_stable(
    rows: Sequence[Mapping[str, object]],
    *,
    model: str,
    axis: str = "all",
    role: str = "all_stable",
) -> Mapping[str, object]:
    matches = [
        row for row in rows if row["model"] == model and row["axis"] == axis and row["role"] == role
    ]
    if len(matches) != 1:
        raise ValueError("stable-control summary row is missing or duplicated")
    return matches[0]


def render_summary(
    taxonomy: Sequence[Mapping[str, object]],
    stable: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]],
) -> str:
    """Render a deterministic interpretation without exposing individual cases."""
    del transitions
    models = sorted({_string(row["model"], "model") for row in taxonomy})
    lines = [
        "# Counterfactual Error Taxonomy",
        "",
        "This is a posthoc descriptive analysis of the frozen controlled diagnostic.",
        "It changes no prompt, prediction, threshold, gate, or comparison eligibility.",
        "",
        "| Model | Stable-admissible false deny | Stable-inadmissible false admit | "
        "Stable overflip | Worst overflip axis |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for model in models:
        admissible = _overall_taxonomy(taxonomy, model=model, role="stable_admissible")
        inadmissible = _overall_taxonomy(taxonomy, model=model, role="stable_inadmissible")
        overall = _overall_stable(stable, model=model)
        axis_rows = [
            row
            for row in stable
            if row["model"] == model and row["axis"] != "all" and row["role"] == "all_stable"
        ]
        worst = max(axis_rows, key=lambda row: float(row["overflip_rate"]))
        lines.append(
            (
                "| {model} | {false_deny:.4f} | {false_admit:.4f} | "
                "{overflip:.4f} | {axis} ({axis_rate:.4f}) |"
            ).format(
                model=model,
                false_deny=float(admissible["false_deny_rate"]),
                false_admit=float(inadmissible["false_admit_rate"]),
                overflip=float(overall["overflip_rate"]),
                axis=worst["axis"],
                axis_rate=float(worst["overflip_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "`false deny` means an explicitly stable-admissible memory was called",
            "inadmissible. `false admit` means an explicitly stable-inadmissible memory",
            "was called admissible. Overflip counts any change in the argmax label between",
            "the two query conditions, including transitions through `unknown`.",
            "",
            "These rates diagnose selectivity after the primary result was frozen. They are",
            "not a new preregistered endpoint and must not be used to tune or rerun providers.",
            "",
            "## Mechanism Read",
            "",
            "The errors are strongly asymmetric. Stable-inadmissible memories are rarely",
            "false-admitted, while stable-admissible memories are often false-denied or",
            "sent to `unknown`. This is a verifier-level over-refusal pattern rather than",
            "evidence of broad failure to detect explicit violations.",
            "",
            "| Model | Lifecycle allow accuracy | Lifecycle block accuracy |",
            "| --- | ---: | ---: |",
        ]
    )
    for model in models:
        lifecycle = _overall_stable(
            stable,
            model=model,
            axis="lifecycle_intent",
        )
        lines.append(
            "| {model} | {allow:.4f} | {block:.4f} |".format(
                model=model,
                allow=float(lifecycle["allow_accuracy"]),
                block=float(lifecycle["block_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "All three models have their largest stable-control overflip on lifecycle",
            "intent. Their lifecycle control accuracy is lower on the history-oriented",
            "`allow` side and reaches 1.0 on the current-state `block` side. Because the",
            "prompt explicitly said not to score topical relevance, this pattern is",
            "consistent with a relevance--admissibility conflation: a current or generally",
            "permitted memory is treated as inadmissible when it is less directly useful to",
            "the changed query. This is a posthoc interpretation of controlled cases, not a",
            "causal diagnosis of model reasoning.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    protocol_path: Path,
    runtime_dir: Path,
    results_dir: Path,
) -> Mapping[str, object]:
    """Verify inputs, aggregate three complete checkpoints, and publish no raw records."""
    protocol = runtime.load_protocol(protocol_path)
    base_manifest_path = results_dir / "manifest.json"
    base_manifest = _mapping(
        json.loads(base_manifest_path.read_text(encoding="utf-8")),
        "base manifest",
    )
    if base_manifest.get("protocol_sha256") != runtime._sha256_file(protocol.path):
        raise ValueError("base result manifest protocol binding drifted")
    eligible = base_manifest.get("comparison_eligible_providers")
    if not isinstance(eligible, list) or not eligible:
        raise ValueError("base result has no comparison-eligible provider")

    taxonomy_rows = []
    stable_rows = []
    transition_rows = []
    checkpoints = []
    for item in eligible:
        identity = _mapping(item, "comparison_eligible_providers[]")
        provider = _string(identity.get("provider"), "provider")
        model_name = _string(identity.get("model"), "model")
        binding = runtime._binding(protocol, provider)
        if binding.model != model_name:
            raise ValueError("eligible model binding drifted")
        predictions, records = runtime._load_provider_predictions(
            protocol,
            binding,
            runtime_dir,
        )
        if len(predictions) != len(protocol.pairs) * len(CONDITIONS):
            raise ValueError("posthoc analysis requires a complete provider bundle")
        completion_path = runtime._completion_path(runtime_dir, binding)
        completion = _mapping(
            json.loads(completion_path.read_text(encoding="utf-8")),
            "completion",
        )
        checkpoint_path = runtime._response_path(runtime_dir, binding)
        if completion.get("checkpoint_sha256") != _sha256(checkpoint_path):
            raise ValueError("completion checkpoint hash drifted")
        model = f"{provider}/{model_name}"
        observations = candidate_observations(protocol.pairs, predictions, model=model)
        stable_observations = stable_pair_observations(observations)
        taxonomy_rows.extend(aggregate_judgments(observations))
        stable_rows.extend(aggregate_stable_pairs(stable_observations))
        transition_rows.extend(aggregate_transitions(stable_observations))
        checkpoints.append(
            {
                "provider": provider,
                "model": model_name,
                "case_count": len(records),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "completion_sha256": _sha256(completion_path),
            }
        )

    output_paths = {
        "error_taxonomy.csv": taxonomy_rows,
        "stable_control_outcomes.csv": stable_rows,
        "overflip_transitions.csv": transition_rows,
    }
    for name, rows in output_paths.items():
        _write_csv(results_dir / name, rows)
    summary_path = results_dir / "error_taxonomy_summary.md"
    _write_text(summary_path, render_summary(taxonomy_rows, stable_rows, transition_rows))

    outputs = [*output_paths, summary_path.name]
    manifest = {
        "schema_version": 1,
        "status": "posthoc_descriptive_error_taxonomy",
        "base_manifest_sha256": _sha256(base_manifest_path),
        "protocol_sha256": runtime._sha256_file(protocol.path),
        "analyzer_sha256": _sha256(Path(__file__)),
        "checkpoints": checkpoints,
        "comparison_eligible_provider_count": len(checkpoints),
        "prompt_or_prediction_modified": False,
        "provider_call_count": 0,
        "individual_case_rows_published": False,
        "query_or_candidate_text_published": False,
        "official_result": False,
        "outputs": [{"path": name, "sha256": _sha256(results_dir / name)} for name in outputs],
    }
    _write_json(results_dir / "posthoc_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=runtime.DEFAULT_PROTOCOL)
    parser.add_argument("--runtime-dir", type=Path, default=runtime.DEFAULT_RUNTIME_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    manifest = analyze(
        protocol_path=args.protocol,
        runtime_dir=args.runtime_dir,
        results_dir=args.results_dir,
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
