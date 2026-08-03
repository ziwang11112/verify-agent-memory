"""Generate paper numbers, tables, and figures from verified normalized evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.verify_evidence import validate_evidence
from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

Row = dict[str, str]
FigureCase = dict[str, Any]


def _load_rows(repository_root: Path) -> list[Row]:
    errors = validate_evidence(repository_root)
    if errors:
        raise ValueError("evidence verification failed: " + "; ".join(errors))
    rows: list[Row] = []
    for path in sorted((repository_root / "evidence" / "normalized").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _load_figure_examples(repository_root: Path) -> list[FigureCase]:
    path = repository_root / "evidence" / "examples" / "retrieval_admissibility_cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("claim_id") != "C1":
        raise ValueError("Figure 1 examples have an unsupported schema or claim")
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, list) or len(source_artifacts) != 2:
        raise ValueError("Figure 1 examples must identify both source artifacts")
    for artifact in source_artifacts:
        digest = artifact.get("sha256") if isinstance(artifact, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Figure 1 examples contain an invalid source hash")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("Figure 1 requires exactly four audited examples")
    required_case_fields = {
        "case_id",
        "source",
        "failure_type",
        "packet_id",
        "query",
        "memories",
        "lesson",
    }
    required_memory_fields = {
        "memory_id",
        "text",
        "relevance",
        "scope",
        "state",
        "prohibited",
        "usable",
    }
    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not required_case_fields <= case.keys():
            raise ValueError("Figure 1 example is missing required fields")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError("Figure 1 example IDs must be unique nonempty strings")
        seen_case_ids.add(case_id)
        memories = case["memories"]
        if not isinstance(memories, list) or len(memories) < 2:
            raise ValueError(f"Figure 1 example {case_id} needs at least two memories")
        for memory in memories:
            if not isinstance(memory, dict) or not required_memory_fields <= memory.keys():
                raise ValueError(f"Figure 1 example {case_id} has a malformed memory")
            if not isinstance(memory["usable"], bool):
                raise ValueError(f"Figure 1 example {case_id} has a non-boolean verdict")
    return cases


def _one(rows: Sequence[Row], **selectors: str) -> Row:
    selected = [
        row for row in rows if all(row.get(field) == value for field, value in selectors.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one evidence row for {selectors}, got {len(selected)}")
    return selected[0]


def _number(row: Mapping[str, str], field: str = "estimate") -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _plain(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _count(row: Mapping[str, str], field: str = "n") -> str:
    value = int(row[field])
    return f"{value:,}".replace(",", "{,}")


def _ci(row: Mapping[str, str], *, signed: bool = True, digits: int = 3) -> str:
    formatter = _signed if signed else _plain
    return (
        f"[{formatter(_number(row, 'ci95_lower'), digits)}, "
        f"{formatter(_number(row, 'ci95_upper'), digits)}]"
    )


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{\ensuremath{{{value}}}}}"


def _write_ascii_lines(path: Path, lines: Sequence[str]) -> None:
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))


def _evidence_rows(rows: Sequence[Row]) -> dict[str, Row]:
    def exposure(source: str, metric: str) -> Row:
        return _one(rows, claim_id="C8", source=source, metric=metric)

    return {
        "reader_a": _one(
            rows,
            claim_id="C2",
            source="gpt-4o-mini-2024-07-18",
            metric="answer_leakage_risk_difference",
        ),
        "reader_b": _one(
            rows,
            claim_id="C2",
            source="gpt-4o-2024-08-06",
            metric="answer_leakage_risk_difference",
        ),
        "reader_a_brier": _one(
            rows,
            claim_id="C2",
            source="gpt-4o-mini-2024-07-18",
            contrast="exposure_augmented_minus_baseline",
            metric="brier_delta",
        ),
        "reader_a_log_loss": _one(
            rows,
            claim_id="C2",
            source="gpt-4o-mini-2024-07-18",
            contrast="exposure_augmented_minus_baseline",
            metric="log_loss_delta",
        ),
        "reader_b_brier": _one(
            rows,
            claim_id="C2",
            source="gpt-4o-2024-08-06",
            contrast="exposure_augmented_minus_baseline",
            metric="brier_delta",
        ),
        "reader_b_log_loss": _one(
            rows,
            claim_id="C2",
            source="gpt-4o-2024-08-06",
            contrast="exposure_augmented_minus_baseline",
            metric="log_loss_delta",
        ),
        "g1_leakage": _one(rows, claim_id="C3", metric="answer_leakage_delta"),
        "g1_utility": _one(rows, claim_id="C3", metric="utility_accuracy_delta"),
        "g1_refusal": _one(rows, claim_id="C3", metric="over_refusal_delta"),
        "smoke_global_recall": _one(
            rows,
            claim_id="C4",
            contrast="global_dense",
            metric="evidence_recall",
        ),
        "smoke_global_contamination": _one(
            rows,
            claim_id="C4",
            contrast="global_dense",
            metric="measured_contamination",
        ),
        "smoke_global_wrong_scope": _one(
            rows,
            claim_id="C4",
            contrast="global_dense",
            metric="wrong_scope_leakage",
        ),
        "smoke_namespace_recall": _one(
            rows,
            claim_id="C4",
            contrast="namespace_dense",
            metric="evidence_recall",
        ),
        "smoke_namespace_contamination": _one(
            rows,
            claim_id="C4",
            contrast="namespace_dense",
            metric="measured_contamination",
        ),
        "smoke_namespace_wrong_scope": _one(
            rows,
            claim_id="C4",
            contrast="namespace_dense",
            metric="wrong_scope_leakage",
        ),
        "global_recall": _one(
            rows,
            claim_id="C5",
            contrast="global_dense",
            metric="evidence_recall",
        ),
        "namespace_recall": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense",
            metric="evidence_recall",
        ),
        "namespace_recall_delta": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_minus_global_dense",
            metric="recall_delta",
        ),
        "namespace_feasible_delta": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_minus_global_dense",
            metric="feasible_rate_delta",
        ),
        "namespace_contamination_delta": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_minus_global_dense",
            metric="conservative_contamination_delta",
        ),
        "namespace_wrong_scope": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense",
            metric="wrong_scope_leakage",
        ),
        "global_feasible": _one(
            rows,
            claim_id="C5",
            contrast="global_dense",
            metric="feasible_rate",
        ),
        "namespace_feasible": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense",
            metric="feasible_rate",
        ),
        "global_penalized_risk": _one(
            rows,
            claim_id="C5",
            contrast="global_dense",
            metric="penalized_contamination_upper",
        ),
        "namespace_penalized_risk": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense",
            metric="penalized_contamination_upper",
        ),
        "global_known_contamination": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="known_contamination",
        ),
        "global_label_coverage": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="label_coverage",
        ),
        "global_lower_bound": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="lower_bound",
        ),
        "global_upper_bound": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="upper_bound",
        ),
        "namespace_known_contamination": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="known_contamination",
        ),
        "namespace_label_coverage": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="label_coverage",
        ),
        "namespace_lower_bound": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="lower_bound",
        ),
        "namespace_upper_bound": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="upper_bound",
        ),
        "threshold_recall_delta": _one(
            rows,
            claim_id="C6",
            contrast="threshold_router_minus_namespace_dense",
            metric="recall_delta",
        ),
        "threshold_contamination_delta": _one(
            rows,
            claim_id="C6",
            contrast="threshold_router_minus_namespace_dense",
            metric="conservative_contamination_delta",
        ),
        "cluster_recall_delta": _one(
            rows,
            claim_id="C6",
            contrast="cluster_router_minus_namespace_dense",
            metric="recall_delta",
        ),
        "cluster_contamination_delta": _one(
            rows,
            claim_id="C6",
            contrast="cluster_router_minus_namespace_dense",
            metric="conservative_contamination_delta",
        ),
        "threshold_fallback_rate": _one(
            rows,
            claim_id="C6",
            contrast="threshold_router",
            metric="fallback_rate",
        ),
        "cluster_route_width": _one(
            rows,
            claim_id="C6",
            contrast="cluster_router",
            metric="mean_route_width",
        ),
        "lifecycle_recall_delta": _one(
            rows,
            claim_id="C7",
            metric="recall_delta",
        ),
        "lifecycle_contamination_delta": _one(
            rows,
            claim_id="C7",
            metric="conservative_contamination_delta",
        ),
        "lifecycle_stale_delta": _one(
            rows,
            claim_id="C7",
            metric="prohibited_stale_exposure_delta",
        ),
        "lifecycle_superseded_delta": _one(
            rows,
            claim_id="C7",
            metric="prohibited_superseded_exposure_delta",
        ),
        "openai_exposure_admissible": exposure("gpt-5.6-sol", "relevant_admissible_effect"),
        "openai_exposure_inadmissible": exposure("gpt-5.6-sol", "relevant_inadmissible_effect"),
        "openai_exposure_gap": exposure("gpt-5.6-sol", "selectivity_gap"),
        "gemini_exposure_admissible": exposure("gemini-3.6-flash", "relevant_admissible_effect"),
        "gemini_exposure_inadmissible": exposure(
            "gemini-3.6-flash", "relevant_inadmissible_effect"
        ),
        "gemini_exposure_gap": exposure("gemini-3.6-flash", "selectivity_gap"),
        "deepseek_exposure_admissible": exposure("deepseek-v4-pro", "relevant_admissible_effect"),
        "deepseek_exposure_inadmissible": exposure(
            "deepseek-v4-pro", "relevant_inadmissible_effect"
        ),
        "deepseek_exposure_gap": exposure("deepseek-v4-pro", "selectivity_gap"),
    }


def _write_numbers(path: Path, selected: Mapping[str, Row]) -> None:
    macros = [
        "% Generated by scripts/build_paper_artifacts.py. Do not edit.",
        _macro("NaturalGroupCount", "87"),
        _macro("NaturalMemoryCount", "182{,}908"),
        _macro("NaturalQueryCount", "3{,}767"),
        _macro("NaturalRouteRows", "33{,}903"),
        _macro("ReaderDiscordantCount", _count(selected["reader_a"])),
        _macro("ReaderPredictionCount", _count(selected["reader_a_brier"])),
        _macro("GOneLeakageCount", _count(selected["g1_leakage"])),
        _macro("GOneUtilityCount", _count(selected["g1_utility"])),
        _macro("ExposureScenarioCount", "16"),
        _macro("ExposurePairsPerReader", "192"),
        _macro(
            "OpenAIExposureAdmissible",
            _plain(_number(selected["openai_exposure_admissible"])),
        ),
        _macro(
            "OpenAIExposureAdmissibleCI",
            _ci(selected["openai_exposure_admissible"], signed=False),
        ),
        _macro(
            "OpenAIExposureInadmissible",
            _plain(_number(selected["openai_exposure_inadmissible"])),
        ),
        _macro(
            "OpenAIExposureInadmissibleCI",
            _ci(selected["openai_exposure_inadmissible"], signed=False),
        ),
        _macro(
            "OpenAIExposureGap",
            _plain(_number(selected["openai_exposure_gap"])),
        ),
        _macro(
            "OpenAIExposureGapCI",
            _ci(selected["openai_exposure_gap"], signed=False),
        ),
        _macro(
            "GeminiExposureAdmissible",
            _plain(_number(selected["gemini_exposure_admissible"])),
        ),
        _macro(
            "GeminiExposureAdmissibleCI",
            _ci(selected["gemini_exposure_admissible"], signed=False),
        ),
        _macro(
            "GeminiExposureInadmissible",
            _plain(_number(selected["gemini_exposure_inadmissible"])),
        ),
        _macro(
            "GeminiExposureInadmissibleCI",
            _ci(selected["gemini_exposure_inadmissible"], signed=False),
        ),
        _macro(
            "GeminiExposureGap",
            _plain(_number(selected["gemini_exposure_gap"])),
        ),
        _macro(
            "GeminiExposureGapCI",
            _ci(selected["gemini_exposure_gap"], signed=False),
        ),
        _macro(
            "DeepSeekExposureAdmissible",
            _plain(_number(selected["deepseek_exposure_admissible"])),
        ),
        _macro(
            "DeepSeekExposureAdmissibleCI",
            _ci(selected["deepseek_exposure_admissible"], signed=False),
        ),
        _macro(
            "DeepSeekExposureInadmissible",
            _plain(_number(selected["deepseek_exposure_inadmissible"])),
        ),
        _macro(
            "DeepSeekExposureInadmissibleCI",
            _ci(selected["deepseek_exposure_inadmissible"], signed=False),
        ),
        _macro(
            "DeepSeekExposureGap",
            _plain(_number(selected["deepseek_exposure_gap"])),
        ),
        _macro(
            "DeepSeekExposureGapCI",
            _ci(selected["deepseek_exposure_gap"], signed=False),
        ),
        _macro(
            "LifecycleExposureCount",
            _count(selected["lifecycle_stale_delta"]),
        ),
        _macro("ReaderARiskDifference", _plain(_number(selected["reader_a"]))),
        _macro(
            "ReaderARiskDifferenceCI",
            _ci(selected["reader_a"], signed=False),
        ),
        _macro("ReaderBRiskDifference", _plain(_number(selected["reader_b"]))),
        _macro(
            "ReaderBRiskDifferenceCI",
            _ci(selected["reader_b"], signed=False),
        ),
        _macro("ReaderABrierDelta", _signed(_number(selected["reader_a_brier"]))),
        _macro("ReaderABrierDeltaCI", _ci(selected["reader_a_brier"])),
        _macro(
            "ReaderALogLossDelta",
            _signed(_number(selected["reader_a_log_loss"])),
        ),
        _macro("ReaderALogLossDeltaCI", _ci(selected["reader_a_log_loss"])),
        _macro("ReaderBBrierDelta", _signed(_number(selected["reader_b_brier"]))),
        _macro("ReaderBBrierDeltaCI", _ci(selected["reader_b_brier"])),
        _macro(
            "ReaderBLogLossDelta",
            _signed(_number(selected["reader_b_log_loss"])),
        ),
        _macro("ReaderBLogLossDeltaCI", _ci(selected["reader_b_log_loss"])),
        _macro("GOneLeakageDelta", _signed(_number(selected["g1_leakage"]))),
        _macro("GOneLeakageDeltaCI", _ci(selected["g1_leakage"])),
        _macro("GOneUtilityDelta", _signed(_number(selected["g1_utility"]))),
        _macro("GOneUtilityDeltaCI", _ci(selected["g1_utility"])),
        _macro("GOneRefusalDelta", _signed(_number(selected["g1_refusal"]))),
        _macro("GOneRefusalDeltaCI", _ci(selected["g1_refusal"])),
        _macro("GOneLeakageReduction", _plain(abs(_number(selected["g1_leakage"])))),
        _macro("GOneUtilityReduction", _plain(abs(_number(selected["g1_utility"])))),
        _macro("GOneRefusalIncrease", _plain(abs(_number(selected["g1_refusal"])))),
        _macro(
            "SmokeGlobalRecall",
            _plain(_number(selected["smoke_global_recall"]), digits=1),
        ),
        _macro(
            "SmokeGlobalContamination",
            _plain(_number(selected["smoke_global_contamination"]), digits=4),
        ),
        _macro(
            "SmokeGlobalWrongScope",
            _plain(_number(selected["smoke_global_wrong_scope"])),
        ),
        _macro(
            "SmokeNamespaceRecall",
            _plain(_number(selected["smoke_namespace_recall"]), digits=1),
        ),
        _macro(
            "SmokeNamespaceContamination",
            _plain(_number(selected["smoke_namespace_contamination"]), digits=4),
        ),
        _macro(
            "SmokeNamespaceWrongScope",
            _plain(_number(selected["smoke_namespace_wrong_scope"]), digits=0),
        ),
        _macro("GlobalDenseRecall", _plain(_number(selected["global_recall"]))),
        _macro("NamespaceDenseRecall", _plain(_number(selected["namespace_recall"]))),
        _macro("GlobalDenseFeasible", _plain(_number(selected["global_feasible"]))),
        _macro(
            "NamespaceDenseFeasible",
            _plain(_number(selected["namespace_feasible"])),
        ),
        _macro(
            "GlobalPenalizedRisk",
            _plain(_number(selected["global_penalized_risk"])),
        ),
        _macro(
            "NamespacePenalizedRisk",
            _plain(_number(selected["namespace_penalized_risk"])),
        ),
        _macro(
            "GlobalKnownContamination",
            _plain(_number(selected["global_known_contamination"])),
        ),
        _macro(
            "NamespaceKnownContamination",
            _plain(_number(selected["namespace_known_contamination"])),
        ),
        _macro(
            "GlobalLabelCoverage",
            _plain(_number(selected["global_label_coverage"])),
        ),
        _macro(
            "NamespaceLabelCoverage",
            _plain(_number(selected["namespace_label_coverage"])),
        ),
        _macro("GlobalLowerBound", _plain(_number(selected["global_lower_bound"]))),
        _macro(
            "NamespaceLowerBound",
            _plain(_number(selected["namespace_lower_bound"])),
        ),
        _macro("GlobalUpperBound", _plain(_number(selected["global_upper_bound"]))),
        _macro(
            "NamespaceUpperBound",
            _plain(_number(selected["namespace_upper_bound"])),
        ),
        _macro(
            "NamespaceRecallDelta",
            _signed(_number(selected["namespace_recall_delta"])),
        ),
        _macro(
            "NamespaceRecallDeltaCI",
            _ci(selected["namespace_recall_delta"]),
        ),
        _macro(
            "NamespaceFeasibleDelta",
            _signed(_number(selected["namespace_feasible_delta"])),
        ),
        _macro(
            "NamespaceFeasibleDeltaCI",
            _ci(selected["namespace_feasible_delta"]),
        ),
        _macro(
            "NamespaceContaminationDelta",
            _signed(_number(selected["namespace_contamination_delta"])),
        ),
        _macro(
            "NamespaceContaminationReduction",
            _plain(abs(_number(selected["namespace_contamination_delta"]))),
        ),
        _macro(
            "NamespaceWrongScopeLeakage",
            _plain(_number(selected["namespace_wrong_scope"])),
        ),
        _macro(
            "NamespaceContaminationDeltaCI",
            _ci(selected["namespace_contamination_delta"]),
        ),
        _macro(
            "ThresholdRecallDelta",
            _signed(_number(selected["threshold_recall_delta"])),
        ),
        _macro(
            "ThresholdRecallLoss",
            _plain(abs(_number(selected["threshold_recall_delta"]))),
        ),
        _macro(
            "ThresholdRecallDeltaCI",
            _ci(selected["threshold_recall_delta"]),
        ),
        _macro(
            "ThresholdContaminationDelta",
            _signed(_number(selected["threshold_contamination_delta"])),
        ),
        _macro(
            "ThresholdContaminationDeltaCI",
            _ci(selected["threshold_contamination_delta"]),
        ),
        _macro(
            "ClusterRecallDelta",
            _signed(_number(selected["cluster_recall_delta"]), digits=4),
        ),
        _macro(
            "ClusterRecallDeltaCI",
            _ci(selected["cluster_recall_delta"], digits=4),
        ),
        _macro(
            "ClusterContaminationDelta",
            _signed(_number(selected["cluster_contamination_delta"]), digits=4),
        ),
        _macro(
            "ClusterContaminationDeltaCI",
            _ci(selected["cluster_contamination_delta"], digits=4),
        ),
        _macro(
            "ThresholdFallbackRate",
            _plain(_number(selected["threshold_fallback_rate"])),
        ),
        _macro(
            "ClusterRouteWidth",
            _plain(_number(selected["cluster_route_width"]), digits=1),
        ),
        _macro(
            "LifecycleRecallDelta",
            _signed(_number(selected["lifecycle_recall_delta"])),
        ),
        _macro(
            "LifecycleRecallDeltaCI",
            _ci(selected["lifecycle_recall_delta"]),
        ),
        _macro(
            "LifecycleContaminationDelta",
            _signed(_number(selected["lifecycle_contamination_delta"])),
        ),
        _macro(
            "LifecycleContaminationReduction",
            _plain(abs(_number(selected["lifecycle_contamination_delta"]))),
        ),
        _macro(
            "LifecycleContaminationDeltaCI",
            _ci(selected["lifecycle_contamination_delta"]),
        ),
        _macro(
            "LifecycleStaleDelta",
            _signed(_number(selected["lifecycle_stale_delta"])),
        ),
        _macro(
            "LifecycleStaleReduction",
            _plain(abs(_number(selected["lifecycle_stale_delta"]))),
        ),
        _macro(
            "LifecycleStaleDeltaCI",
            _ci(selected["lifecycle_stale_delta"]),
        ),
        _macro(
            "LifecycleSupersededDelta",
            _signed(_number(selected["lifecycle_superseded_delta"])),
        ),
        _macro(
            "LifecycleSupersededReduction",
            _plain(abs(_number(selected["lifecycle_superseded_delta"]))),
        ),
        _macro(
            "LifecycleSupersededDeltaCI",
            _ci(selected["lifecycle_superseded_delta"]),
        ),
    ]
    _write_ascii_lines(path, macros)


def _write_main_table(path: Path, selected: Mapping[str, Row]) -> None:
    entries = [
        (
            "Paired exposure",
            "GPT-5.6 selectivity gap",
            _plain(_number(selected["openai_exposure_gap"])),
            _ci(selected["openai_exposure_gap"], signed=False),
            "C8",
        ),
        (
            "Paired exposure",
            "Gemini 3.6 selectivity gap",
            _plain(_number(selected["gemini_exposure_gap"])),
            _ci(selected["gemini_exposure_gap"], signed=False),
            "C8",
        ),
        (
            "Paired exposure",
            "DeepSeek V4 selectivity gap",
            _plain(_number(selected["deepseek_exposure_gap"])),
            _ci(selected["deepseek_exposure_gap"], signed=False),
            "C8",
        ),
        (
            "Paired exposure",
            "DeepSeek inadmissible effect",
            _plain(_number(selected["deepseek_exposure_inadmissible"])),
            _ci(selected["deepseek_exposure_inadmissible"], signed=False),
            "C8",
        ),
        (
            "G1 vs. G0",
            "Answer leakage",
            _signed(_number(selected["g1_leakage"])),
            _ci(selected["g1_leakage"]),
            "C3",
        ),
        (
            "Namespace vs. global",
            "Evidence recall",
            _signed(_number(selected["namespace_recall_delta"])),
            _ci(selected["namespace_recall_delta"]),
            "C5",
        ),
        (
            "Namespace vs. global",
            "Feasible rate",
            _signed(_number(selected["namespace_feasible_delta"])),
            _ci(selected["namespace_feasible_delta"]),
            "C5",
        ),
        (
            "Namespace vs. global",
            "Penalized conservative risk",
            _signed(_number(selected["namespace_contamination_delta"])),
            _ci(selected["namespace_contamination_delta"]),
            "C5",
        ),
        (
            "Lifecycle upper bound",
            "Penalized conservative risk",
            _signed(_number(selected["lifecycle_contamination_delta"])),
            _ci(selected["lifecycle_contamination_delta"]),
            "C7",
        ),
    ]
    lines = [
        r"\begin{tabular}{llrrc}",
        r"\toprule",
        r"Contrast & Metric & Estimate & 95\% CI & Claim \\",
        r"\midrule",
    ]
    for contrast, metric, estimate, interval, claim_id in entries:
        lines.append(f"{contrast} & {metric} & {estimate} & {interval} & {claim_id} \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write_ascii_lines(path, lines)


def _natural_arm_row(rows: Sequence[Row], arm: str, metric: str) -> Row:
    return _one(
        rows,
        claim_id="C5",
        contrast=arm,
        metric=metric,
    )


NATURAL_ARMS = (
    ("global_bm25", "Global BM25", "BM25", "global", "s"),
    ("global_dense", "Global dense", "Dense", "global", "s"),
    ("global_bm25_dense_rrf", "Global BM25+dense RRF", "RRF", "global", "s"),
    ("global_recency_dense", "Global recency dense", "Recency", "global", "s"),
    ("namespace_dense", "Namespace dense", "Namespace dense", "namespace", "o"),
    (
        "namespace_current_only",
        "Namespace current-only",
        "Current-only",
        "namespace",
        "o",
    ),
    (
        "released_intent_lifecycle_upper_bound",
        "Released lifecycle UB",
        "Lifecycle UB",
        "upper_bound",
        "*",
    ),
    ("threshold_router", "Threshold router", "Threshold", "router", "D"),
    ("cluster_router", "Cluster router", "Cluster", "router", "D"),
)


def _write_full_arm_table(path: Path, rows: Sequence[Row]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Arm & Recall & Feasible & Penalized risk & Candidates \\",
        r"\midrule",
    ]
    for arm, label, _short_label, _family, _marker in NATURAL_ARMS:
        recall = _natural_arm_row(rows, arm, "evidence_recall")
        feasible = _natural_arm_row(rows, arm, "feasible_rate")
        risk = _natural_arm_row(rows, arm, "penalized_contamination_upper")
        candidates = _natural_arm_row(rows, arm, "mean_candidates_scored")
        lines.append(
            f"{label} & {_plain(_number(recall))}"
            f" & {_plain(_number(feasible))}"
            f" & {_plain(_number(risk))}"
            f" & {_plain(_number(candidates), digits=0)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write_ascii_lines(path, lines)


def _write_matched_prefix_table(path: Path, selected: Mapping[str, Row]) -> None:
    entries = (
        (
            "Global dense",
            selected["global_known_contamination"],
            selected["global_label_coverage"],
            selected["global_lower_bound"],
            selected["global_upper_bound"],
        ),
        (
            "Namespace dense",
            selected["namespace_known_contamination"],
            selected["namespace_label_coverage"],
            selected["namespace_lower_bound"],
            selected["namespace_upper_bound"],
        ),
    )
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Support & Known contam. & Coverage & Lower & Upper \\",
        r"\midrule",
    ]
    for label, known, coverage, lower, upper in entries:
        lines.append(
            f"{label} & {_plain(_number(known))}"
            f" & {_plain(_number(coverage))}"
            f" & {_plain(_number(lower))}"
            f" & {_plain(_number(upper))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write_ascii_lines(path, lines)


def _write_smoke_table(path: Path, selected: Mapping[str, Row]) -> None:
    entries = (
        (
            "Global dense",
            selected["smoke_global_recall"],
            selected["smoke_global_wrong_scope"],
            selected["smoke_global_contamination"],
        ),
        (
            "Namespace dense",
            selected["smoke_namespace_recall"],
            selected["smoke_namespace_wrong_scope"],
            selected["smoke_namespace_contamination"],
        ),
    )
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Support & Evidence recall & Wrong-scope leakage & Contamination \\",
        r"\midrule",
    ]
    for label, recall, wrong_scope, contamination in entries:
        lines.append(
            f"{label} & {_plain(_number(recall), digits=1)}"
            f" & {_plain(_number(wrong_scope))}"
            f" & {_plain(_number(contamination), digits=4)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write_ascii_lines(path, lines)


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "verify-agent-memory",
        }
    )


def _save_figure(figure: object, stem: Path) -> None:
    metadata = {
        "Creator": "verify-agent-memory",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    svg_path = stem.with_suffix(".svg")
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "verify-agent-memory", "Date": None},
    )
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(
        stem.with_suffix(".png"),
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "verify-agent-memory"},
    )


def _write_pipeline_figure(path: Path, cases: Sequence[FigureCase]) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(7.1, 4.25))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    keep_color = "#2E8B57"
    drop_color = "#B64342"
    neutral = "#59616B"
    case_colors = {
        "wrong_namespace_market": "#4C78A8",
        "superseded_lisbon_date": "#D9822B",
        "forgotten_phone_detail": "#B6532E",
        "historical_metformin_timeline": "#7A5195",
    }

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        facecolor: str = "#FFFFFF",
        edgecolor: str = "#C1C6CD",
        linewidth: float = 0.7,
        radius: float = 0.009,
    ) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        axis.add_patch(patch)
        return patch

    def verdict(memory: Mapping[str, Any]) -> str:
        if memory["usable"]:
            if memory["state"] in {"stale", "superseded"}:
                return "KEEP - old state is needed for this history query"
            return "KEEP - scope, policy, and lifecycle pass"
        reasons: list[str] = []
        if memory["scope"] == "disallowed":
            reasons.append("wrong namespace")
        if memory["state"] in {"stale", "superseded"}:
            reasons.append("lifecycle conflict")
        if memory["prohibited"] is True:
            reasons.append("policy block")
        if not reasons:
            reasons.append("not usable for this query")
        return "DROP - " + " + ".join(reasons)

    def render_case(
        case: FigureCase,
        *,
        panel: str,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        accent = case_colors[str(case["case_id"])]
        box(x, y, width, height, edgecolor=accent, linewidth=0.9)
        axis.add_patch(
            FancyBboxPatch(
                (x, y + height - 0.053),
                width,
                0.053,
                boxstyle="round,pad=0.006,rounding_size=0.009",
                linewidth=0,
                facecolor=accent,
                alpha=0.11,
            )
        )
        axis.text(
            x + 0.012,
            y + height - 0.026,
            f"({panel}) {case['failure_type']}",
            ha="left",
            va="center",
            weight="bold",
            color=accent,
            fontsize=6.5,
        )
        source = str(case["source"])
        axis.text(
            x + width - 0.012,
            y + height - 0.026,
            source,
            ha="right",
            va="center",
            color=neutral,
            fontsize=4.7,
            weight="bold",
        )

        query_y = y + height - 0.127
        box(
            x + 0.012,
            query_y,
            width - 0.024,
            0.057,
            facecolor="#F5F7FA",
            edgecolor="#D8DCE2",
            linewidth=0.55,
            radius=0.006,
        )
        axis.text(
            x + 0.023,
            query_y + 0.029,
            "Q",
            ha="left",
            va="center",
            weight="bold",
            color="#315F93",
            fontsize=5.6,
        )
        axis.text(
            x + 0.046,
            query_y + 0.029,
            textwrap.fill(str(case["query"]), width=58),
            ha="left",
            va="center",
            color="#252A30",
            fontsize=5.2,
            linespacing=1.08,
        )

        memories = case["memories"]
        row_top = y + height - 0.151
        row_bottom = y + 0.057
        gap = 0.007
        row_height = (row_top - row_bottom - gap * (len(memories) - 1)) / len(memories)
        for index, memory in enumerate(memories):
            row_y = row_top - (index + 1) * row_height - index * gap
            usable = bool(memory["usable"])
            color = keep_color if usable else drop_color
            facecolor = "#EDF7F1" if usable else "#FCEFED"
            box(
                x + 0.012,
                row_y,
                width - 0.024,
                row_height,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=0.55,
                radius=0.005,
            )
            badge_width = 0.043
            box(
                x + 0.020,
                row_y + 0.010,
                badge_width,
                max(0.025, row_height - 0.020),
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                radius=0.004,
            )
            axis.text(
                x + 0.020 + badge_width / 2,
                row_y + row_height / 2,
                "USE" if usable else "DROP",
                ha="center",
                va="center",
                color="#FFFFFF",
                weight="bold",
                fontsize=4.4,
            )
            text_x = x + 0.073
            if len(memories) > 2:
                axis.text(
                    text_x,
                    row_y + row_height / 2,
                    str(memory["text"]),
                    ha="left",
                    va="center",
                    color="#24292F",
                    fontsize=4.4,
                )
                history_label = (
                    "old state | kept for history"
                    if memory["state"] in {"stale", "superseded"}
                    else "current | checks pass"
                )
                axis.text(
                    x + width - 0.015,
                    row_y + row_height / 2,
                    history_label,
                    ha="right",
                    va="center",
                    color=color,
                    fontsize=3.75,
                    weight="bold",
                )
            else:
                axis.text(
                    text_x,
                    row_y + row_height - 0.012,
                    textwrap.fill(str(memory["text"]), width=52),
                    ha="left",
                    va="top",
                    color="#24292F",
                    fontsize=4.9,
                    linespacing=1.06,
                )
                axis.text(
                    text_x,
                    row_y + 0.010,
                    verdict(memory),
                    ha="left",
                    va="bottom",
                    color=color,
                    fontsize=4.25,
                    weight="bold",
                )

        axis.text(
            x + 0.014,
            y + 0.023,
            str(case["lesson"]),
            ha="left",
            va="center",
            color=accent,
            fontsize=4.55,
            weight="bold",
        )

    case_by_id = {str(case["case_id"]): case for case in cases}
    panel_specs = (
        ("wrong_namespace_market", "a", 0.015, 0.565),
        ("superseded_lisbon_date", "b", 0.510, 0.565),
        ("forgotten_phone_detail", "c", 0.015, 0.175),
        ("historical_metformin_timeline", "d", 0.510, 0.175),
    )
    for case_id, panel, x, y in panel_specs:
        render_case(case_by_id[case_id], panel=panel, x=x, y=y, width=0.475, height=0.365)

    box(
        0.015,
        0.025,
        0.970,
        0.105,
        facecolor="#F5F7FA",
        edgecolor="#98A1AB",
        linewidth=0.8,
    )
    axis.text(
        0.029,
        0.105,
        "VERIFICATION LAYER",
        ha="left",
        va="center",
        color="#343A40",
        weight="bold",
        fontsize=5.7,
    )
    method_boxes = (
        (0.150, 0.055, 0.130, "ranked candidate IDs", "#FFFFFF", "#7C838C"),
        (0.340, 0.055, 0.080, "scope", "#EEF4FA", "#4C78A8"),
        (0.435, 0.055, 0.080, "policy", "#F5F0F8", "#7A5195"),
        (0.530, 0.055, 0.130, "lifecycle (q, t)", "#FFF5E5", "#D9822B"),
        (0.720, 0.055, 0.245, "USE / DROP + reason-coded trace", "#EDF7F1", "#2E8B57"),
    )
    for x, y, width, label, facecolor, edgecolor in method_boxes:
        box(
            x,
            y,
            width,
            0.045,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.65,
            radius=0.005,
        )
        axis.text(
            x + width / 2,
            y + 0.0225,
            label,
            ha="center",
            va="center",
            color="#30363C",
            fontsize=4.65,
            weight="bold",
        )
    for source_x, target_x in ((0.285, 0.333), (0.665, 0.713)):
        axis.add_patch(
            FancyArrowPatch(
                (source_x, 0.0775),
                (target_x, 0.0775),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.75,
                color="#70777F",
            )
        )
    axis.text(
        0.150,
        0.038,
        "Known violation: exclude | unresolved: report unknown | "
        "old state: keep only when query-compatible",
        ha="left",
        va="center",
        color=neutral,
        fontsize=4.15,
    )

    axis.text(
        0.985,
        0.975,
        "Abridged audited public-source examples",
        ha="right",
        va="top",
        color="#747B83",
        fontsize=4.5,
    )

    figure.tight_layout(pad=0.15)
    _save_figure(figure, path)
    plt.close(figure)


def _errorbar(
    axis: object,
    rows: Sequence[Row],
    labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    estimates = [_number(row) for row in rows]
    lower = [
        estimate - _number(row, "ci95_lower") for estimate, row in zip(estimates, rows, strict=True)
    ]
    upper = [
        _number(row, "ci95_upper") - estimate for estimate, row in zip(estimates, rows, strict=True)
    ]
    positions = list(range(len(rows)))
    for position, estimate, low, high, color in zip(
        positions,
        estimates,
        lower,
        upper,
        colors,
        strict=True,
    ):
        axis.errorbar(
            estimate,
            position,
            xerr=[[low], [high]],
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.2,
            color=color,
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)


def _write_evidence_figure(path: Path, selected: Mapping[str, Row]) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    reader_a = "#087F8C"
    reader_b = "#C65D2E"
    neutral = "#7C838C"
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), layout="constrained")
    risk_axis, predictive_axis, tradeoff_axis = axes

    _errorbar(
        risk_axis,
        [selected["reader_a"], selected["reader_b"]],
        ["Reader A", "Reader B"],
        [reader_a, reader_b],
    )
    risk_axis.set_title("(a) Exposure and disclosure", loc="left", weight="bold")
    risk_axis.set_xlabel("Exposed minus unexposed disclosure risk")
    risk_axis.set_xlim(0.34, 0.60)
    risk_axis.grid(False)
    risk_axis.text(
        0.98,
        0.94,
        "721 discordant checkpoints",
        transform=risk_axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=neutral,
    )
    for position, key, color in (
        (0, "reader_a", reader_a),
        (1, "reader_b", reader_b),
    ):
        estimate = _number(selected[key])
        risk_axis.text(
            estimate + 0.012,
            position,
            f"{estimate:.3f}",
            color=color,
            va="center",
            fontsize=6.2,
            weight="bold",
        )

    _errorbar(
        predictive_axis,
        [
            selected["reader_a_brier"],
            selected["reader_b_brier"],
            selected["reader_a_log_loss"],
            selected["reader_b_log_loss"],
        ],
        ["Brier / A", "Brier / B", "Log loss / A", "Log loss / B"],
        [reader_a, reader_b, reader_a, reader_b],
    )
    predictive_axis.set_title("(b) Predictive value of exposure", loc="left", weight="bold")
    predictive_axis.set_xlabel("Delta vs. no-exposure model (lower is better)")
    predictive_axis.set_xlim(-0.085, 0.005)
    predictive_axis.grid(False)
    predictive_axis.axhline(1.5, color="#D9DDE2", linewidth=0.7)
    predictive_axis.text(
        0.98,
        0.94,
        "4,470 held-out rows",
        transform=predictive_axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=neutral,
    )

    _errorbar(
        tradeoff_axis,
        [
            selected["g1_leakage"],
            selected["g1_utility"],
            selected["g1_refusal"],
        ],
        ["Answer leakage", "Bounded utility", "Over-refusal"],
        ["#2E8B57", "#B64342", "#D9822B"],
    )
    tradeoff_axis.set_title("(c) Safety and utility trade-off", loc="left", weight="bold")
    tradeoff_axis.set_xlabel("G1 minus G0")
    tradeoff_axis.set_xlim(-0.30, 0.38)
    tradeoff_axis.grid(False)
    for position, key, color in (
        (0, "g1_leakage", "#2E8B57"),
        (1, "g1_utility", "#B64342"),
        (2, "g1_refusal", "#D9822B"),
    ):
        estimate = _number(selected[key])
        offset = -0.018 if estimate < 0 else 0.018
        tradeoff_axis.text(
            estimate + offset,
            position,
            f"{estimate:+.3f}",
            ha="right" if estimate < 0 else "left",
            va="center",
            color=color,
            fontsize=6.0,
            weight="bold",
        )

    _save_figure(figure, path)
    plt.close(figure)


def _write_retrieval_figure(
    path: Path,
    rows: Sequence[Row],
    selected: Mapping[str, Row],
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    namespace_color = "#2E8B57"
    threshold_color = "#D9822B"
    cluster_color = "#4C78A8"
    lifecycle_color = "#7A5195"
    global_color = "#858B93"
    current_only_color = "#79A989"
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.25), layout="constrained")

    arm_colors = {
        "global_bm25": global_color,
        "global_dense": global_color,
        "global_bm25_dense_rrf": global_color,
        "global_recency_dense": global_color,
        "namespace_dense": namespace_color,
        "namespace_current_only": current_only_color,
        "released_intent_lifecycle_upper_bound": lifecycle_color,
        "threshold_router": threshold_color,
        "cluster_router": cluster_color,
    }
    label_positions = {
        "global_bm25": (105000, 0.448),
        "global_dense": (105000, 0.713),
        "global_bm25_dense_rrf": (105000, 0.685),
        "global_recency_dense": (105000, 0.748),
        "namespace_dense": (1900, 0.886),
        "namespace_current_only": (1900, 0.815),
        "released_intent_lifecycle_upper_bound": (1900, 0.903),
        "threshold_router": (1900, 0.836),
        "cluster_router": (1900, 0.861),
    }
    axes[0, 0].axvspan(1100, 2600, color="#EAF4EE", alpha=0.75, zorder=-3)
    axes[0, 0].axvspan(70000, 115000, color="#F0F1F3", alpha=0.85, zorder=-3)
    for arm, _label, short_label, _family, marker in NATURAL_ARMS:
        recall = _number(_natural_arm_row(rows, arm, "evidence_recall"))
        candidates = _number(_natural_arm_row(rows, arm, "mean_candidates_scored"))
        size = 68 if marker == "*" else 30
        axes[0, 0].scatter(
            candidates,
            recall,
            color=arm_colors[arm],
            marker=marker,
            s=size,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axes[0, 0].annotate(
            short_label,
            xy=(candidates, recall),
            xytext=label_positions[arm],
            textcoords="data",
            fontsize=5.1,
            color=arm_colors[arm],
            weight="bold" if arm in {"namespace_dense", "global_dense"} else "normal",
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": arm_colors[arm],
                "linewidth": 0.45,
                "shrinkA": 1.5,
                "shrinkB": 2.5,
            },
        )
    global_candidates = _number(_natural_arm_row(rows, "global_dense", "mean_candidates_scored"))
    namespace_candidates = _number(
        _natural_arm_row(rows, "namespace_dense", "mean_candidates_scored")
    )
    candidate_ratio = global_candidates / namespace_candidates
    axes[0, 0].annotate(
        "",
        xy=(namespace_candidates, 0.49),
        xytext=(global_candidates, 0.49),
        arrowprops={"arrowstyle": "<->", "color": "#5B6168", "linewidth": 0.8},
    )
    axes[0, 0].text(
        math.sqrt(global_candidates * namespace_candidates),
        0.505,
        f"{candidate_ratio:.0f}x fewer candidates",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#4F565E",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlim(900, 130000)
    axes[0, 0].set_ylim(0.40, 0.91)
    axes[0, 0].set_xlabel("Mean candidates scored (log scale)")
    axes[0, 0].set_ylabel("Evidence recall")
    axes[0, 0].set_title("(a) Support size, recall, and all nine arms", loc="left", weight="bold")
    axes[0, 0].grid(False)

    _errorbar(
        axes[0, 1],
        [
            selected["namespace_recall_delta"],
            selected["namespace_feasible_delta"],
            selected["namespace_contamination_delta"],
        ],
        ["Evidence recall", "Feasible rate", "Penalized risk"],
        [namespace_color, namespace_color, namespace_color],
    )
    axes[0, 1].set_title("(b) Trusted namespace main effects", loc="left", weight="bold")
    axes[0, 1].set_xlabel("Namespace minus global")
    axes[0, 1].set_xlim(-0.065, 0.28)
    axes[0, 1].grid(False)
    for position, key in enumerate(
        ("namespace_recall_delta", "namespace_feasible_delta", "namespace_contamination_delta")
    ):
        estimate = _number(selected[key])
        axes[0, 1].text(
            estimate + (0.011 if estimate >= 0 else -0.011),
            position,
            f"{estimate:+.3f}",
            ha="left" if estimate >= 0 else "right",
            va="center",
            fontsize=5.8,
            color=namespace_color,
            weight="bold",
        )
    bound_rows = (
        (
            "Global dense",
            selected["global_known_contamination"],
            selected["global_label_coverage"],
            selected["global_lower_bound"],
            selected["global_upper_bound"],
            global_color,
        ),
        (
            "Namespace dense",
            selected["namespace_known_contamination"],
            selected["namespace_label_coverage"],
            selected["namespace_lower_bound"],
            selected["namespace_upper_bound"],
            namespace_color,
        ),
    )
    for position, (_label, known, coverage, lower, upper, color) in enumerate(bound_rows):
        lower_value = _number(lower)
        upper_value = _number(upper)
        known_value = _number(known)
        axes[1, 0].plot(
            (lower_value, upper_value),
            (position, position),
            color=color,
            linewidth=8,
            alpha=0.24,
            solid_capstyle="butt",
        )
        axes[1, 0].plot(
            (lower_value, upper_value),
            (position, position),
            color=color,
            linewidth=1.0,
            marker="|",
            markersize=8,
        )
        axes[1, 0].scatter(
            known_value,
            position,
            color=color,
            marker="o",
            s=28,
            zorder=3,
        )
        axes[1, 0].text(
            0.98,
            position,
            f"coverage {_number(coverage):.3f}",
            ha="right",
            va="center",
            fontsize=5.4,
            color=color,
        )
    axes[1, 0].set_yticks((0, 1), ("Global dense", "Namespace dense"))
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, 1.0)
    axes[1, 0].set_xlabel("Contamination fraction within feasible matched prefixes")
    axes[1, 0].set_title(
        "(c) Unresolved labels keep the upper bound high", loc="left", weight="bold"
    )
    axes[1, 0].grid(False)
    effects = (
        ("Threshold recall", selected["threshold_recall_delta"], False, threshold_color),
        (
            "Threshold risk",
            selected["threshold_contamination_delta"],
            True,
            threshold_color,
        ),
        ("Cluster recall", selected["cluster_recall_delta"], False, cluster_color),
        ("Cluster risk", selected["cluster_contamination_delta"], True, cluster_color),
        ("Lifecycle recall", selected["lifecycle_recall_delta"], False, lifecycle_color),
        (
            "Lifecycle risk",
            selected["lifecycle_contamination_delta"],
            True,
            lifecycle_color,
        ),
        ("Stale exposure", selected["lifecycle_stale_delta"], True, lifecycle_color),
        (
            "Superseded exposure",
            selected["lifecycle_superseded_delta"],
            True,
            lifecycle_color,
        ),
    )
    effect_positions = (0, 1, 2.7, 3.7, 5.4, 6.4, 7.4, 8.4)
    for position, (_label, row, lower_is_better, color) in zip(
        effect_positions,
        effects,
        strict=True,
    ):
        raw_estimate = _number(row)
        raw_lower = _number(row, "ci95_lower")
        raw_upper = _number(row, "ci95_upper")
        if lower_is_better:
            estimate = -raw_estimate
            lower = -raw_upper
            upper = -raw_lower
        else:
            estimate = raw_estimate
            lower = raw_lower
            upper = raw_upper
        axes[1, 1].errorbar(
            estimate,
            position,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            markersize=4.2,
            capsize=2.3,
            linewidth=1.0,
            color=color,
        )
    axes[1, 1].set_yticks(effect_positions, [effect[0] for effect in effects])
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(0, color="#666C73", linewidth=0.8, linestyle="--")
    axes[1, 1].axhline(1.85, color="#D9DDE2", linewidth=0.6)
    axes[1, 1].axhline(4.55, color="#D9DDE2", linewidth=0.6)
    axes[1, 1].set_xlim(-0.032, 0.014)
    axes[1, 1].set_xlabel("Benefit-oriented change (positive is favorable)")
    axes[1, 1].set_title("(d) Add-ons over namespace dense", loc="left", weight="bold")
    axes[1, 1].grid(False)
    axes[1, 1].text(
        -0.031,
        4.75,
        "Threshold fallback 96.5% | Cluster width 1.0",
        ha="left",
        va="center",
        fontsize=5.2,
        color="#5B6168",
    )

    _save_figure(figure, path)
    plt.close(figure)


def _write_manifest(
    repository_root: Path,
    output_root: Path,
    outputs: Sequence[Path],
) -> None:
    evidence_inputs = sorted(
        [
            *(repository_root / "evidence" / "normalized").glob("*.csv"),
            *(repository_root / "evidence" / "examples").glob("*.json"),
        ],
        key=lambda path: path.as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "builder": "scripts/build_paper_artifacts.py",
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": [
            {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in evidence_inputs
        ],
        "outputs": [
            {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(outputs)
        ],
    }
    (output_root / "artifact_manifest.json").write_bytes(canonical_json_bytes(manifest))


def build(repository_root: Path) -> tuple[Path, ...]:
    rows = _load_rows(repository_root)
    figure_examples = _load_figure_examples(repository_root)
    selected = _evidence_rows(rows)
    output_root = repository_root / "paper" / "generated"
    output_root.mkdir(parents=True, exist_ok=True)
    numbers_path = output_root / "paper_numbers.tex"
    main_table_path = output_root / "main_results.tex"
    full_arm_table_path = output_root / "full_arm_results.tex"
    matched_prefix_table_path = output_root / "matched_prefix_diagnostics.tex"
    smoke_table_path = output_root / "mechanism_smoke.tex"
    outputs = (
        numbers_path,
        main_table_path,
        full_arm_table_path,
        matched_prefix_table_path,
        smoke_table_path,
        output_root / "verification_pipeline.pdf",
        output_root / "verification_pipeline.png",
        output_root / "verification_pipeline.svg",
        output_root / "evidence_summary.pdf",
        output_root / "evidence_summary.png",
        output_root / "evidence_summary.svg",
        output_root / "retrieval_results.pdf",
        output_root / "retrieval_results.png",
        output_root / "retrieval_results.svg",
    )
    _write_numbers(numbers_path, selected)
    _write_main_table(main_table_path, selected)
    _write_full_arm_table(full_arm_table_path, rows)
    _write_matched_prefix_table(matched_prefix_table_path, selected)
    _write_smoke_table(smoke_table_path, selected)
    _write_pipeline_figure(output_root / "verification_pipeline", figure_examples)
    _write_evidence_figure(output_root / "evidence_summary", selected)
    _write_retrieval_figure(output_root / "retrieval_results", rows, selected)
    _write_manifest(repository_root, output_root, outputs)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build(args.repository_root.resolve())
    print(f"generated {len(outputs)} paper artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
