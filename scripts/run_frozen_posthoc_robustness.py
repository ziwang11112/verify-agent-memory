"""Build verifier operating curves and few-scenario robustness from frozen scores."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.run_inferred_admissibility_experiment import (
    _load_response_records,
    _predictions_from_records,
    _response_path,
    load_protocol,
)
from verify_agent_memory.inferred_admissibility import (
    FilterSetting,
    aggregate_route_scores,
    filter_decision_metrics,
    load_cases,
    route_keys,
    score_filtered_route,
    sha256_file,
)
from verify_agent_memory.support_controls import (
    exact_sign_flip_pvalue,
    leave_one_out_means,
    scenario_selectivity,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "posthoc_operating_curves_protocol.json"
INFERENCE_PROTOCOL_PATH = ROOT / "experiments" / "inferred_admissibility_protocol.json"
DEFAULT_CASES = ROOT / "tmp" / "inferred_admissibility" / "cases.jsonl"
DEFAULT_RESPONSES = ROOT / "tmp" / "inferred_admissibility_canonical" / "primary_first_committed"
PAIR_SCORES_PATH = ROOT / "results" / "counterfactual_exposure" / "pair_scores.csv"
SELECTED_PATH = ROOT / "results" / "inferred_admissibility" / "selected_thresholds.csv"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output {path}")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"inconsistent fields for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_curve_protocol() -> Mapping[str, object]:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != (
        "posthoc-operating-curves-and-cluster-robustness-v1"
    ):
        raise ValueError("posthoc operating-curve protocol identity drifted")
    thresholds = tuple(float(value) for value in raw["threshold_grid"])
    if thresholds != tuple(sorted(set(thresholds))) or thresholds[0] != 0 or thresholds[-1] != 1:
        raise ValueError("operating-curve threshold grid drifted")
    return raw


def _selected_thresholds() -> dict[str, float]:
    with SELECTED_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["provider"]: float(row["violation_threshold"])
        for row in rows
        if row["arm"] == "text_inferred"
    }


def _equal_stratum_route_metrics(
    cases: Sequence[object], scores: Sequence[object]
) -> dict[str, object]:
    if len(cases) != len(scores):
        raise ValueError("cases and route scores must align")
    strata = sorted({f"{case.source}/{case.released_query_intent}" for case in cases})
    rows = []
    for stratum in strata:
        selected = [
            score
            for case, score in zip(cases, scores, strict=True)
            if f"{case.source}/{case.released_query_intent}" == stratum
        ]
        rows.append(aggregate_route_scores(selected))
    fields = (
        "evidence_recall",
        "feasible_rate",
        "penalized_admissibility_upper_risk",
        "conditional_admissibility_upper_risk",
        "known_admissibility_violation_rate",
        "mean_route_width",
    )
    return {
        field: (
            sum(float(row[field]) for row in rows if row[field] is not None)
            / sum(row[field] is not None for row in rows)
            if any(row[field] is not None for row in rows)
            else None
        )
        for field in fields
    }


def _operating_curve_rows(
    curve_protocol: Mapping[str, object],
    *,
    cases_path: Path,
    response_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    inference_protocol = load_protocol(INFERENCE_PROTOCOL_PATH)
    if (
        sha256_file(INFERENCE_PROTOCOL_PATH)
        != curve_protocol["inferred_admissibility_protocol_sha256"]
    ):
        raise ValueError("inferred-admissibility protocol hash drifted")
    cases = load_cases(cases_path)
    analysis = tuple(case for case in cases if case.role == "analysis")
    selected = _selected_thresholds()
    expected_hashes = dict(curve_protocol["complete_response_sha256"])
    rows: list[dict[str, object]] = []
    response_hashes: dict[str, str] = {}
    for binding in inference_protocol.providers:
        if binding.provider not in expected_hashes:
            continue
        response_path = _response_path(response_dir, binding)
        observed_hash = sha256_file(response_path)
        if observed_hash != expected_hashes[binding.provider]:
            raise ValueError(f"{binding.provider} response hash drifted")
        response_hashes[binding.provider] = observed_hash
        records = _load_response_records(response_path)
        predictions = _predictions_from_records(records, cases, binding, inference_protocol)
        for threshold in curve_protocol["threshold_grid"]:
            setting = FilterSetting(
                violation_threshold=float(threshold),
                unknown_threshold=None,
                known_violation_precision=None,
                known_violation_recall=0.0,
                required_anchor_false_deny_rate=0.0,
                dropped_unknown_gold=0,
            )
            decisions = filter_decision_metrics(analysis, predictions, setting)
            route_scores = []
            for case in analysis:
                keys = route_keys(
                    case,
                    arm="text_inferred",
                    prediction=predictions[case.case_id],
                    setting=setting,
                )
                route_scores.append(
                    score_filtered_route(
                        case,
                        keys,
                        target_recall=float(curve_protocol["target_recall"]),
                    )
                )
            route = _equal_stratum_route_metrics(analysis, route_scores)
            rows.append(
                {
                    "provider": binding.provider,
                    "model": binding.model,
                    "threshold": threshold,
                    "dev_selected_threshold": threshold == selected[binding.provider],
                    "candidate_count": decisions["candidate_count"],
                    "retained_candidate_fraction": (
                        1 - float(decisions["dropped_count"]) / float(decisions["candidate_count"])
                    ),
                    "violation_precision": decisions["violation_precision"],
                    "violation_recall": decisions["violation_recall"],
                    "required_anchor_false_deny_rate": decisions["required_anchor_false_deny_rate"],
                    **route,
                }
            )
    return rows, response_hashes


def _scenario_rows(curve_protocol: Mapping[str, object]) -> list[dict[str, object]]:
    if sha256_file(PAIR_SCORES_PATH) != curve_protocol["counterfactual_pair_scores_sha256"]:
        raise ValueError("counterfactual pair-score hash drifted")
    with PAIR_SCORES_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    providers = sorted({(row["provider"], row["model"]) for row in rows})
    output = []
    for provider, model in providers:
        selected = [row for row in rows if row["provider"] == provider and row["model"] == model]
        deltas = scenario_selectivity(selected)
        if len(deltas) != int(curve_protocol["counterfactual_scenario_count"]):
            raise ValueError("counterfactual scenario count drifted")
        loo = leave_one_out_means(deltas)
        values = tuple(deltas.values())
        output.append(
            {
                "provider": provider,
                "model": model,
                "scenario_count": len(values),
                "mean_selectivity_gap": sum(values) / len(values),
                "leave_one_scenario_out_min": min(loo.values()),
                "leave_one_scenario_out_max": max(loo.values()),
                "scenarios_positive": sum(value > 0 for value in values),
                "scenarios_zero": sum(value == 0 for value in values),
                "scenarios_negative": sum(value < 0 for value in values),
                "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(values),
            }
        )
    return output


def _write_readme(
    output_dir: Path,
    curve_rows: Sequence[Mapping[str, object]],
    scenario_rows: Sequence[Mapping[str, object]],
) -> None:
    selected = [row for row in curve_rows if row["dev_selected_threshold"]]
    lines = [
        "# Frozen Operating Curves and Few-Scenario Robustness",
        "",
        "The verifier table sweeps a fixed 0.00--1.00 threshold grid on the frozen 72-case ",
        "analysis partition. The original calibration-selected threshold is marked but never ",
        "changed. Curves expose precision--recall, retained-candidate coverage, required-anchor ",
        "false denial, evidence recall, feasible rate, and matched-recall risk.",
        "Candidate precision, recall, and false denial are analysis-candidate micro averages; ",
        "route outcomes are equal-stratum macro averages, matching the primary diagnostic.",
        "",
        "## Dev-selected verifier points",
        "",
        "| Reader | Threshold | Precision | Violation recall | False deny | Route risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        precision = row["violation_precision"]
        precision_text = "NA" if precision is None else f"{float(precision):.4f}"
        lines.append(
            f"| `{row['model']}` | {float(row['threshold']):.2f} | {precision_text} | "
            f"{float(row['violation_recall']):.4f} | "
            f"{float(row['required_anchor_false_deny_rate']):.4f} | "
            f"{float(row['penalized_admissibility_upper_risk']):.4f} |"
        )
    for provider in sorted({str(row["provider"]) for row in curve_rows}):
        safe = [
            row
            for row in curve_rows
            if row["provider"] == provider and float(row["required_anchor_false_deny_rate"]) <= 0.01
        ]
        if safe:
            best_recall = max(float(row["violation_recall"]) for row in safe)
            lines.append(
                f"{provider} has {len(safe)} grid point(s) at or below 1% anchor false denial; "
                f"the best violation recall there is {best_recall:.4f}."
            )
        else:
            lines.append(f"{provider} has no grid point at or below 1% anchor false denial.")
    lines.extend(
        [
            "",
            "## Leave-one-scenario-out selectivity",
            "",
            "| Reader | Mean gap | LOO range | Exact sign-flip p |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in scenario_rows:
        lines.append(
            f"| `{row['model']}` | {float(row['mean_selectivity_gap']):.4f} | "
            f"[{float(row['leave_one_scenario_out_min']):.4f}, "
            f"{float(row['leave_one_scenario_out_max']):.4f}] | "
            f"{float(row['exact_two_sided_sign_flip_p']):.6f} |"
        )
    lines.extend(
        [
            "",
            "These are post-hoc robustness analyses of frozen outputs, not new model runs or ",
            "official benchmark results. Providers remain separate. No response, prompt, raw ",
            "text, case ID, candidate ID, or scenario ID is published here.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _run(cases_path: Path, response_dir: Path, output_dir: Path) -> None:
    protocol = _load_curve_protocol()
    curve_rows, response_hashes = _operating_curve_rows(
        protocol,
        cases_path=cases_path,
        response_dir=response_dir,
    )
    scenario_rows = _scenario_rows(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "verifier_operating_curve.csv"
    scenario_path = output_dir / "counterfactual_scenario_robustness.csv"
    _write_csv(curve_path, curve_rows)
    _write_csv(scenario_path, scenario_rows)
    _write_readme(output_dir, curve_rows, scenario_rows)
    readme_path = output_dir / "README.md"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "analysis": "posthoc-operating-curves-and-cluster-robustness-v1",
            "analysis_commit": head,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "response_sha256": response_hashes,
            "case_bundle_sha256": sha256_file(cases_path),
            "counterfactual_pair_scores_sha256": sha256_file(PAIR_SCORES_PATH),
            "outputs": {
                curve_path.name: {"sha256": sha256_file(curve_path), "rows": len(curve_rows)},
                scenario_path.name: {
                    "sha256": sha256_file(scenario_path),
                    "rows": len(scenario_rows),
                },
                readme_path.name: {"sha256": sha256_file(readme_path)},
            },
            "contains_case_candidate_scenario_or_group_ids": False,
            "contains_raw_text_prompts_or_responses": False,
            "evaluation_retuning": False,
            "model_pooling": False,
            "provider_calls": 0,
            "reader_calls": 0,
            "judge_calls": 0,
            "paid_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "curve_rows": len(curve_rows),
                "scenario_rows": len(scenario_rows),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / "posthoc_robustness")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _run(args.cases.resolve(), args.response_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
