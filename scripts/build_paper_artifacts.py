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

    def opus_exposure(metric: str) -> Row:
        return _one(rows, claim_id="C12", source="claude-opus-5", metric=metric)

    selected = {
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
            metric="measured_non_usable_rate",
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
            metric="measured_non_usable_rate",
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
        "global_candidates": _one(
            rows,
            claim_id="C5",
            contrast="global_dense",
            metric="mean_candidates_scored",
        ),
        "namespace_candidates": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense",
            metric="mean_candidates_scored",
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
            metric="penalized_non_usable_upper_risk_delta",
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
            metric="penalized_non_usable_upper_risk",
        ),
        "namespace_penalized_risk": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense",
            metric="penalized_non_usable_upper_risk",
        ),
        "global_known_contamination": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="known_non_usable_rate",
        ),
        "global_label_coverage": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="non_usable_label_coverage",
        ),
        "global_lower_bound": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="non_usable_lower_bound",
        ),
        "global_upper_bound": _one(
            rows,
            claim_id="C5",
            contrast="global_dense_matched_prefix",
            metric="non_usable_upper_bound",
        ),
        "namespace_known_contamination": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="known_non_usable_rate",
        ),
        "namespace_label_coverage": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="non_usable_label_coverage",
        ),
        "namespace_lower_bound": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="non_usable_lower_bound",
        ),
        "namespace_upper_bound": _one(
            rows,
            claim_id="C5",
            contrast="namespace_dense_matched_prefix",
            metric="non_usable_upper_bound",
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
            metric="penalized_non_usable_upper_risk_delta",
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
            metric="penalized_non_usable_upper_risk_delta",
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
            metric="penalized_non_usable_upper_risk_delta",
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
        "opus_exposure_admissible": opus_exposure("relevant_admissible_effect"),
        "opus_exposure_inadmissible": opus_exposure("relevant_inadmissible_effect"),
        "opus_exposure_gap": opus_exposure("selectivity_gap"),
    }
    for top_k in (10, 20, 50, 100):
        for metric in (
            "global_evidence_recall",
            "namespace_evidence_recall",
            "global_feasible_rate",
            "namespace_feasible_rate",
            "global_penalized_admissibility_upper_risk",
            "namespace_penalized_admissibility_upper_risk",
            "evidence_recall_delta",
            "feasible_rate_delta",
            "penalized_admissibility_upper_risk_delta",
            "infeasibility_risk_component_delta",
            "admissibility_conditional_risk_component_delta",
        ):
            selected[f"c9_k{top_k}_{metric}"] = _one(
                rows,
                claim_id="C9",
                contrast=f"top_k_{top_k}",
                metric=metric,
            )
    for contrast in ("policy_only", "lifecycle_only", "governance_v2"):
        for metric in (
            "evidence_recall_delta",
            "feasible_rate_delta",
            "penalized_admissibility_upper_risk_delta",
        ):
            selected[f"c10_{contrast}_{metric}"] = _one(
                rows,
                claim_id="C10",
                contrast=contrast,
                metric=metric,
            )
    for contrast in (
        "namespace_false_deny",
        "namespace_missing",
        "namespace_false_allow",
        "namespace_swap",
        "policy_false_deny",
        "policy_false_allow",
        "policy_missing",
        "lifecycle_false_stale",
    ):
        selected[f"c10_{contrast}_last"] = _one(
            rows,
            claim_id="C10",
            contrast=contrast,
            metric="last_observed_dominating_rate",
        )
        first = [
            row
            for row in rows
            if row.get("claim_id") == "C10"
            and row.get("contrast") == contrast
            and row.get("metric") == "first_observed_non_dominating_rate"
        ]
        if first:
            if len(first) != 1:
                raise ValueError(f"duplicate C10 break-even row for {contrast}")
            selected[f"c10_{contrast}_first"] = first[0]
    for contrast in ("false_allow_at_0_5", "namespace_swap_at_0_2"):
        for metric in (
            "evidence_recall",
            "feasible_rate",
            "penalized_admissibility_upper_risk",
            "matched_prefix_wrong_scope_exposure_rate",
            "candidates_scored",
        ):
            selected[f"c10_{contrast}_{metric}"] = _one(
                rows,
                claim_id="C10",
                contrast=contrast,
                metric=metric,
            )
    for contrast in ("released_oracle", "openai_text_inferred", "gemini_text_inferred"):
        for metric in (
            "evidence_recall_delta",
            "feasible_rate_delta",
            "penalized_admissibility_upper_risk_delta",
        ):
            selected[f"c11_{contrast}_{metric}"] = _one(
                rows,
                claim_id="C11",
                contrast=contrast,
                metric=metric,
            )
    for contrast in ("openai_text_inferred", "gemini_text_inferred"):
        for metric in (
            "roc_auc",
            "violation_precision",
            "violation_recall",
            "required_anchor_false_deny_rate",
        ):
            selected[f"c11_{contrast}_{metric}"] = _one(
                rows,
                claim_id="C11",
                contrast=contrast,
                metric=metric,
            )
    for contrast in ("gpt56_controlled", "gemini_controlled", "deepseek_controlled"):
        for metric in (
            "strict_focal_pair_consistency",
            "stable_control_overflip_rate",
            "stable_admissible_false_deny_rate",
            "stable_inadmissible_false_admit_rate",
        ):
            selected[f"c11_{contrast}_{metric}"] = _one(
                rows,
                claim_id="C11",
                contrast=contrast,
                metric=metric,
            )
    for source, reader in (
        ("deepseek-v4-pro", "deepseek"),
        ("gemini-3.6-flash", "gemini"),
        ("gpt-5.6-luna", "gpt_luna"),
    ):
        for metric in (
            "evidence_recall",
            "feasible",
            "penalized_admissibility_upper_risk",
            "answer_correct",
            "answer_quality",
            "over_refusal",
            "protected_disclosure",
            "stale_disclosure",
        ):
            selected[f"c13_{reader}_namespace_{metric}"] = _one(
                rows,
                claim_id="C13",
                source=source,
                contrast="namespace_dense_minus_global_dense",
                metric=metric,
            )
        for metric in ("penalized_admissibility_upper_risk", "answer_correct"):
            selected[f"c13_{reader}_policy_{metric}"] = _one(
                rows,
                claim_id="C13",
                source=source,
                contrast="namespace_policy_gate_minus_namespace_dense",
                metric=metric,
            )
        if source != "gpt-5.6-luna":
            for metric in ("penalized_admissibility_upper_risk", "answer_correct"):
                selected[f"c13_{reader}_text_{metric}"] = _one(
                    rows,
                    claim_id="C13",
                    source=source,
                    contrast="namespace_text_verifier_minus_namespace_dense",
                    metric=metric,
                )
    return selected


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
            "OpusExposureAdmissible",
            _plain(_number(selected["opus_exposure_admissible"])),
        ),
        _macro(
            "OpusExposureAdmissibleCI",
            _ci(selected["opus_exposure_admissible"], signed=False),
        ),
        _macro(
            "OpusExposureInadmissible",
            _plain(_number(selected["opus_exposure_inadmissible"])),
        ),
        _macro(
            "OpusExposureInadmissibleCI",
            _ci(selected["opus_exposure_inadmissible"], signed=False),
        ),
        _macro(
            "OpusExposureGap",
            _plain(_number(selected["opus_exposure_gap"])),
        ),
        _macro(
            "OpusExposureGapCI",
            _ci(selected["opus_exposure_gap"], signed=False),
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
            "SmokeGlobalNonUsableRate",
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
            "SmokeNamespaceNonUsableRate",
            _plain(_number(selected["smoke_namespace_contamination"]), digits=4),
        ),
        _macro(
            "SmokeNamespaceWrongScope",
            _plain(_number(selected["smoke_namespace_wrong_scope"]), digits=0),
        ),
        _macro("GlobalDenseRecall", _plain(_number(selected["global_recall"]))),
        _macro("NamespaceDenseRecall", _plain(_number(selected["namespace_recall"]))),
        _macro(
            "GlobalDenseCandidates",
            f"{_number(selected['global_candidates']):,.0f}".replace(",", "{,}"),
        ),
        _macro(
            "NamespaceDenseCandidates",
            f"{_number(selected['namespace_candidates']):,.0f}".replace(",", "{,}"),
        ),
        _macro(
            "CandidateWorkRatio",
            _plain(
                _number(selected["global_candidates"]) / _number(selected["namespace_candidates"]),
                digits=1,
            ),
        ),
        _macro("GlobalDenseFeasible", _plain(_number(selected["global_feasible"]))),
        _macro(
            "NamespaceDenseFeasible",
            _plain(_number(selected["namespace_feasible"])),
        ),
        _macro(
            "GlobalPenalizedNonUsableRisk",
            _plain(_number(selected["global_penalized_risk"])),
        ),
        _macro(
            "NamespacePenalizedNonUsableRisk",
            _plain(_number(selected["namespace_penalized_risk"])),
        ),
        _macro(
            "GlobalKnownNonUsableRate",
            _plain(_number(selected["global_known_contamination"])),
        ),
        _macro(
            "NamespaceKnownNonUsableRate",
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
            "NamespaceNonUsableRiskDelta",
            _signed(_number(selected["namespace_contamination_delta"])),
        ),
        _macro(
            "NamespaceNonUsableRiskReduction",
            _plain(abs(_number(selected["namespace_contamination_delta"]))),
        ),
        _macro(
            "NamespaceWrongScopeLeakage",
            _plain(_number(selected["namespace_wrong_scope"])),
        ),
        _macro(
            "NamespaceNonUsableRiskDeltaCI",
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
            "ThresholdNonUsableRiskDelta",
            _signed(_number(selected["threshold_contamination_delta"])),
        ),
        _macro(
            "ThresholdNonUsableRiskDeltaCI",
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
            "ClusterNonUsableRiskDelta",
            _signed(_number(selected["cluster_contamination_delta"]), digits=4),
        ),
        _macro(
            "ClusterNonUsableRiskDeltaCI",
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
            "HistoricalVOneRecallDelta",
            _signed(_number(selected["lifecycle_recall_delta"])),
        ),
        _macro(
            "HistoricalVOneRecallDeltaCI",
            _ci(selected["lifecycle_recall_delta"]),
        ),
        _macro(
            "HistoricalVOneNonUsableRiskDelta",
            _signed(_number(selected["lifecycle_contamination_delta"])),
        ),
        _macro(
            "HistoricalVOneNonUsableRiskReduction",
            _plain(abs(_number(selected["lifecycle_contamination_delta"]))),
        ),
        _macro(
            "HistoricalVOneNonUsableRiskDeltaCI",
            _ci(selected["lifecycle_contamination_delta"]),
        ),
        _macro(
            "HistoricalVOneStaleExposureDelta",
            _signed(_number(selected["lifecycle_stale_delta"])),
        ),
        _macro(
            "HistoricalVOneStaleExposureReduction",
            _plain(abs(_number(selected["lifecycle_stale_delta"]))),
        ),
        _macro(
            "HistoricalVOneStaleExposureDeltaCI",
            _ci(selected["lifecycle_stale_delta"]),
        ),
        _macro(
            "HistoricalVOneSupersededExposureDelta",
            _signed(_number(selected["lifecycle_superseded_delta"])),
        ),
        _macro(
            "HistoricalVOneSupersededExposureReduction",
            _plain(abs(_number(selected["lifecycle_superseded_delta"]))),
        ),
        _macro(
            "HistoricalVOneSupersededExposureDeltaCI",
            _ci(selected["lifecycle_superseded_delta"]),
        ),
    ]
    for top_k, label in ((20, "Twenty"), (100, "Hundred")):
        for metric, suffix in (
            ("global_evidence_recall", "GlobalRecall"),
            ("namespace_evidence_recall", "NamespaceRecall"),
            ("global_feasible_rate", "GlobalFeasible"),
            ("namespace_feasible_rate", "NamespaceFeasible"),
            ("global_penalized_admissibility_upper_risk", "GlobalAdmRisk"),
            ("namespace_penalized_admissibility_upper_risk", "NamespaceAdmRisk"),
        ):
            macros.append(
                _macro(f"Top{label}{suffix}", _plain(_number(selected[f"c9_k{top_k}_{metric}"])))
            )
        for metric, suffix in (
            ("evidence_recall_delta", "RecallDelta"),
            ("feasible_rate_delta", "FeasibleDelta"),
            ("penalized_admissibility_upper_risk_delta", "AdmRiskDelta"),
            ("infeasibility_risk_component_delta", "InfeasibilityDelta"),
            (
                "admissibility_conditional_risk_component_delta",
                "ConditionalAdmDelta",
            ),
        ):
            row = selected[f"c9_k{top_k}_{metric}"]
            macros.extend(
                (
                    _macro(f"Top{label}{suffix}", _signed(_number(row))),
                    _macro(f"Top{label}{suffix}CI", _ci(row)),
                )
            )
    for contrast, label in (
        ("policy_only", "PolicyOnly"),
        ("lifecycle_only", "LifecycleOnly"),
        ("governance_v2", "GovernanceVTwo"),
    ):
        for metric, suffix in (
            ("evidence_recall_delta", "RecallDelta"),
            ("feasible_rate_delta", "FeasibleDelta"),
            ("penalized_admissibility_upper_risk_delta", "AdmRiskDelta"),
        ):
            row = selected[f"c10_{contrast}_{metric}"]
            macros.extend(
                (
                    _macro(f"{label}{suffix}", _signed(_number(row))),
                    _macro(f"{label}{suffix}CI", _ci(row)),
                )
            )
    for contrast, label in (
        ("namespace_false_deny", "NamespaceFalseDeny"),
        ("namespace_missing", "NamespaceMissing"),
        ("namespace_false_allow", "NamespaceFalseAllow"),
        ("namespace_swap", "NamespaceSwap"),
        ("policy_false_deny", "PolicyFalseDeny"),
        ("policy_false_allow", "PolicyFalseAllow"),
        ("policy_missing", "PolicyMissing"),
        ("lifecycle_false_stale", "LifecycleFalseStale"),
    ):
        macros.append(
            _macro(
                f"{label}LastDominating",
                _plain(_number(selected[f"c10_{contrast}_last"]), digits=2),
            )
        )
        first_key = f"c10_{contrast}_first"
        if first_key in selected:
            macros.append(
                _macro(
                    f"{label}FirstNonDominating",
                    _plain(_number(selected[first_key]), digits=2),
                )
            )
    for contrast, label in (
        ("false_allow_at_0_5", "FalseAllowHalf"),
        ("namespace_swap_at_0_2", "NamespaceSwapTwenty"),
    ):
        for metric, suffix in (
            ("evidence_recall", "Recall"),
            ("feasible_rate", "Feasible"),
            ("penalized_admissibility_upper_risk", "AdmRisk"),
            ("matched_prefix_wrong_scope_exposure_rate", "WrongScope"),
            ("candidates_scored", "Candidates"),
        ):
            digits = 0 if metric == "candidates_scored" else 3
            macros.append(
                _macro(
                    f"{label}{suffix}",
                    _plain(_number(selected[f"c10_{contrast}_{metric}"]), digits=digits),
                )
            )
    for contrast, label in (
        ("released_oracle", "ReleasedOracle"),
        ("openai_text_inferred", "OpenAIInferred"),
        ("gemini_text_inferred", "GeminiInferred"),
    ):
        for metric, suffix in (
            ("evidence_recall_delta", "RecallDelta"),
            ("feasible_rate_delta", "FeasibleDelta"),
            ("penalized_admissibility_upper_risk_delta", "AdmRiskDelta"),
        ):
            row = selected[f"c11_{contrast}_{metric}"]
            macros.extend(
                (
                    _macro(f"{label}{suffix}", _signed(_number(row))),
                    _macro(f"{label}{suffix}CI", _ci(row)),
                )
            )
    for contrast, label in (
        ("openai_text_inferred", "OpenAIInferred"),
        ("gemini_text_inferred", "GeminiInferred"),
    ):
        for metric, suffix in (
            ("roc_auc", "RocAuc"),
            ("violation_precision", "ViolationPrecision"),
            ("violation_recall", "ViolationRecall"),
            ("required_anchor_false_deny_rate", "AnchorFalseDeny"),
        ):
            macros.append(
                _macro(f"{label}{suffix}", _plain(_number(selected[f"c11_{contrast}_{metric}"])))
            )
    for contrast, label in (
        ("gpt56_controlled", "GPTControlled"),
        ("gemini_controlled", "GeminiControlled"),
        ("deepseek_controlled", "DeepSeekControlled"),
    ):
        for metric, suffix in (
            ("strict_focal_pair_consistency", "FocalConsistency"),
            ("stable_control_overflip_rate", "Overflip"),
        ):
            row = selected[f"c11_{contrast}_{metric}"]
            macros.extend(
                (
                    _macro(f"{label}{suffix}", _plain(_number(row))),
                    _macro(f"{label}{suffix}CI", _ci(row, signed=False)),
                )
            )
        for metric, suffix in (
            ("stable_admissible_false_deny_rate", "StableFalseDeny"),
            ("stable_inadmissible_false_admit_rate", "StableFalseAdmit"),
        ):
            macros.append(
                _macro(f"{label}{suffix}", _plain(_number(selected[f"c11_{contrast}_{metric}"])))
            )
    for reader, label in (
        ("deepseek", "DeepSeekNatural"),
        ("gemini", "GeminiNatural"),
        ("gpt_luna", "GPTLunaNatural"),
    ):
        for metric, suffix in (
            ("answer_correct", "AccuracyDelta"),
            ("answer_quality", "QualityDelta"),
            ("over_refusal", "OverRefusalDelta"),
            ("protected_disclosure", "ProtectedDisclosureDelta"),
            ("stale_disclosure", "StaleDisclosureDelta"),
        ):
            row = selected[f"c13_{reader}_namespace_{metric}"]
            macros.extend(
                (
                    _macro(f"{label}{suffix}", _signed(_number(row))),
                    _macro(f"{label}{suffix}CI", _ci(row)),
                )
            )
        policy = selected[f"c13_{reader}_policy_answer_correct"]
        macros.extend(
            (
                _macro(f"{label}PolicyAccuracyDelta", _signed(_number(policy))),
                _macro(f"{label}PolicyAccuracyDeltaCI", _ci(policy)),
            )
        )
    for metric, suffix in (
        ("evidence_recall", "RecallDelta"),
        ("feasible", "FeasibleDelta"),
        ("penalized_admissibility_upper_risk", "AdmRiskDelta"),
    ):
        row = selected[f"c13_deepseek_namespace_{metric}"]
        macros.extend(
            (
                _macro(f"NaturalEndToEnd{suffix}", _signed(_number(row))),
                _macro(f"NaturalEndToEnd{suffix}CI", _ci(row)),
            )
        )
    policy_risk = selected["c13_deepseek_policy_penalized_admissibility_upper_risk"]
    text_risk = selected["c13_deepseek_text_penalized_admissibility_upper_risk"]
    macros.extend(
        (
            _macro("NaturalPolicyAdmRiskDelta", _signed(_number(policy_risk))),
            _macro("NaturalPolicyAdmRiskDeltaCI", _ci(policy_risk)),
            _macro("NaturalTextVerifierAdmRiskDelta", _signed(_number(text_risk), digits=4)),
            _macro("NaturalTextVerifierAdmRiskDeltaCI", _ci(text_risk, digits=4)),
            _macro("NaturalEndToEndCaseCount", "1{,}523"),
        )
    )
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
            "Reader replication",
            "Claude Opus 5 selectivity gap",
            _plain(_number(selected["opus_exposure_gap"])),
            _ci(selected["opus_exposure_gap"], signed=False),
            "C12",
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
            "Penalized non-usable upper risk",
            _signed(_number(selected["namespace_contamination_delta"])),
            _ci(selected["namespace_contamination_delta"]),
            "C5",
        ),
        (
            "Historical released-field v1",
            "Penalized non-usable upper risk",
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


def _write_natural_end_to_end_table(path: Path, selected: Mapping[str, Row]) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Reader & $\Delta$ answer accuracy & $\Delta$ answer quality & $\Delta$ over-refusal \\",
        r"\midrule",
    ]
    for reader, label in (
        ("deepseek", "DeepSeek V4 Pro"),
        ("gemini", "Gemini 3.6 Flash"),
        ("gpt_luna", r"GPT-5.6 Luna$^{\dagger}$"),
    ):
        cells = []
        for metric in ("answer_correct", "answer_quality", "over_refusal"):
            row = selected[f"c13_{reader}_namespace_{metric}"]
            cells.append(f"{_signed(_number(row))} {_ci(row)}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
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
        "Query-agnostic current-only",
        "Current-only",
        "namespace",
        "o",
    ),
    (
        "released_intent_lifecycle_upper_bound",
        "Historical released-field v1",
        "Released-field v1",
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
        r"Arm & Recall & Feasible & \shortstack{Penalized non-usable\\upper risk} & Candidates \\",
        r"\midrule",
    ]
    for arm, label, _short_label, _family, _marker in NATURAL_ARMS:
        recall = _natural_arm_row(rows, arm, "evidence_recall")
        feasible = _natural_arm_row(rows, arm, "feasible_rate")
        risk = _natural_arm_row(rows, arm, "penalized_non_usable_upper_risk")
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
        r"Support & Known non-usable & Coverage & Lower & Upper \\",
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
        r"Support & Evidence recall & Wrong-scope leakage & Non-usable fraction \\",
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


def _write_verification_examples_figure(path: Path, cases: Sequence[FigureCase]) -> None:
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
        "Violation: exclude | evaluator unresolved -> verifier may return unknown | "
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


def _write_pipeline_figure_legacy(
    path: Path,
    cases: Sequence[FigureCase],
    selected: Mapping[str, Row],
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(7.2, 3.55))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    colors = {
        "allowed": "#248553",
        "blocked": "#C43D3D",
        "trusted": "#2166A5",
        "neutral": "#52606D",
        "dark": "#20262D",
        "line": "#C9D0D7",
        "pale_blue": "#EEF5FA",
        "pale_green": "#EAF5EE",
        "pale_red": "#FBEDEE",
        "pale_gray": "#F4F6F8",
        "pale_gold": "#FFF6E7",
        "gold": "#B86A16",
    }

    def rounded_box(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        facecolor: str = "#FFFFFF",
        edgecolor: str | None = None,
        linewidth: float = 0.8,
        radius: float = 0.006,
    ) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.005,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor or colors["line"],
            linewidth=linewidth,
        )
        axis.add_patch(patch)
        return patch

    def pill(
        x: float,
        y: float,
        width: float,
        label: str,
        *,
        facecolor: str,
        edgecolor: str,
        text_color: str,
    ) -> None:
        rounded_box(
            x,
            y,
            width,
            0.036,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.55,
            radius=0.010,
        )
        axis.text(
            x + width / 2,
            y + 0.018,
            label,
            ha="center",
            va="center",
            fontsize=4.7,
            weight="bold",
            color=text_color,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = "#7B848C",
        width: float = 0.9,
        linestyle: str | tuple[int, tuple[int, ...]] = "solid",
        connectionstyle: str = "arc3",
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=width,
                linestyle=linestyle,
                color=color,
                connectionstyle=connectionstyle,
                shrinkA=1,
                shrinkB=1,
            )
        )

    case_by_id = {str(case["case_id"]): case for case in cases}

    scope_case = case_by_id["wrong_namespace_market"]
    scope_blocked = next(
        memory for memory in scope_case["memories"] if memory["scope"] == "disallowed"
    )
    scope_allowed = next(memory for memory in scope_case["memories"] if memory["usable"])

    lifecycle_case = case_by_id["superseded_lisbon_date"]
    lifecycle_blocked = next(
        memory for memory in lifecycle_case["memories"] if memory["state"] == "superseded"
    )
    lifecycle_allowed = next(
        memory for memory in lifecycle_case["memories"] if memory["state"] == "current"
    )

    policy_case = case_by_id["forgotten_phone_detail"]
    policy_blocked = next(
        memory
        for memory in policy_case["memories"]
        if memory["prohibited"] and not memory["usable"]
    )
    policy_allowed = next(memory for memory in policy_case["memories"] if memory["usable"])

    recall_delta = 100 * _number(selected["c9_k20_evidence_recall_delta"])
    risk_delta = 100 * _number(selected["c9_k20_penalized_admissibility_upper_risk_delta"])
    overflip_values = [
        100 * _number(selected[f"c11_{model}_stable_control_overflip_rate"])
        for model in ("gpt56_controlled", "gemini_controlled", "deepseek_controlled")
    ]
    exposure_delta = 100 * _number(selected["deepseek_exposure_inadmissible"])

    # Panel a: three audited failure types using one fixed visual grammar.
    axis.text(0.014, 0.965, "a", fontsize=8, weight="bold", va="top")
    axis.text(
        0.040,
        0.965,
        "Semantic overlap is not enough to admit a memory",
        fontsize=7.8,
        weight="bold",
        va="top",
        color=colors["dark"],
    )
    axis.text(
        0.040,
        0.915,
        "Three audited examples; the row grammar is query -> retrieved memory -> check -> decision",
        fontsize=4.7,
        color=colors["neutral"],
    )

    for x, label in (
        (0.095, "QUERY"),
        (0.343, "RETRIEVED MEMORY"),
        (0.555, "FAILED CHECK"),
        (0.657, "BEFORE PROMPT"),
    ):
        axis.text(
            x,
            0.865,
            label,
            ha="center",
            fontsize=4.8,
            weight="bold",
            color=colors["neutral"],
        )
    example_rows = (
        (
            0.605,
            "SCOPE",
            "RHELM",
            colors["trusted"],
            scope_case,
            scope_blocked,
            scope_allowed,
            "N = 0",
            "wrong principal",
            None,
        ),
        (
            0.335,
            "LIFECYCLE",
            "MemOps",
            colors["gold"],
            lifecycle_case,
            lifecycle_blocked,
            lifecycle_allowed,
            "L = 0",
            "superseded now",
            "history intent: same record may pass",
        ),
        (
            0.065,
            "POLICY",
            "MemOps",
            "#7A5195",
            policy_case,
            policy_blocked,
            policy_allowed,
            "P = 0",
            "explicitly forgotten",
            None,
        ),
    )
    for (
        y,
        axis_label,
        source,
        axis_color,
        case,
        blocked_memory,
        allowed_memory,
        check_label,
        check_reason,
        intent_note,
    ) in example_rows:
        rounded_box(
            0.018,
            y,
            0.680,
            0.225,
            facecolor="#FFFFFF",
            edgecolor=colors["line"],
            linewidth=0.65,
        )
        axis.plot([0.024, 0.024], [y + 0.014, y + 0.211], color=axis_color, lw=3.0)
        axis.text(
            0.037,
            y + 0.190,
            axis_label,
            fontsize=5.3,
            weight="bold",
            color=axis_color,
        )
        axis.text(
            0.111,
            y + 0.190,
            source,
            fontsize=4.3,
            color=colors["neutral"],
        )
        axis.text(
            0.037,
            y + 0.112,
            textwrap.fill(str(case["query"]), width=27),
            fontsize=4.9,
            color=colors["dark"],
            va="center",
            linespacing=1.12,
        )

        rounded_box(
            0.195,
            y + 0.025,
            0.300,
            0.175,
            facecolor=colors["pale_gray"],
            edgecolor=colors["line"],
            linewidth=0.55,
        )
        axis.text(
            0.209,
            y + 0.175,
            "TOPICAL MATCH",
            fontsize=4.4,
            weight="bold",
            color=colors["trusted"],
        )
        axis.text(
            0.209,
            y + 0.120,
            textwrap.fill(f"“{blocked_memory['text']}”", width=45),
            fontsize=4.7,
            color=colors["dark"],
            va="center",
            linespacing=1.08,
        )
        rounded_box(
            0.207,
            y + 0.037,
            0.276,
            0.037,
            facecolor=colors["pale_green"],
            edgecolor="#C7DFD0",
            linewidth=0.4,
            radius=0.004,
        )
        eligible_text = textwrap.shorten(
            str(allowed_memory["text"]),
            width=47,
            placeholder="...",
        )
        axis.text(
            0.218,
            y + 0.055,
            f"eligible alternative: {eligible_text}",
            fontsize=3.8,
            color=colors["allowed"],
            va="center",
        )

        arrow((0.498, y + 0.112), (0.512, y + 0.112), color=colors["blocked"])
        rounded_box(
            0.516,
            y + 0.045,
            0.092,
            0.135,
            facecolor=colors["pale_red"],
            edgecolor=colors["blocked"],
            linewidth=0.75,
        )
        axis.text(
            0.562,
            y + 0.137,
            check_label,
            ha="center",
            fontsize=5.6,
            weight="bold",
            color=colors["blocked"],
        )
        axis.text(
            0.562,
            y + 0.087,
            textwrap.fill(check_reason, width=16),
            ha="center",
            va="center",
            fontsize=4.2,
            color=colors["neutral"],
        )

        arrow((0.611, y + 0.112), (0.624, y + 0.112), color=colors["blocked"])
        rounded_box(
            0.628,
            y + 0.067,
            0.056,
            0.090,
            facecolor=colors["pale_red"],
            edgecolor=colors["blocked"],
            linewidth=0.85,
        )
        axis.text(
            0.656,
            y + 0.112,
            "DROP",
            ha="center",
            va="center",
            fontsize=5.2,
            weight="bold",
            color=colors["blocked"],
        )
        if intent_note is not None:
            rounded_box(
                0.516,
                y + 0.009,
                0.168,
                0.030,
                facecolor=colors["pale_blue"],
                edgecolor="#BFD4E8",
                linewidth=0.4,
                radius=0.004,
            )
            axis.text(
                0.600,
                y + 0.024,
                intent_note,
                ha="center",
                va="center",
                fontsize=3.5,
                color=colors["trusted"],
            )

    # Panel b: the observable path and the three distinct empirical tests.
    axis.plot([0.715, 0.715], [0.04, 0.94], color="#D7DCE1", lw=0.8)
    axis.text(0.735, 0.965, "b", fontsize=8, weight="bold", va="top")
    axis.text(
        0.762,
        0.965,
        "Verify before exposure",
        fontsize=7.6,
        weight="bold",
        va="top",
        color=colors["dark"],
    )

    path_stages = (
        (0.733, 0.850, 0.058, "RETRIEVE", colors["pale_gray"], colors["neutral"]),
        (0.800, 0.850, 0.055, "VERIFY", colors["pale_blue"], colors["trusted"]),
        (0.864, 0.850, 0.055, "EXPOSE", colors["pale_green"], colors["allowed"]),
        (0.928, 0.850, 0.058, "DISCLOSE", colors["pale_red"], colors["blocked"]),
    )
    for x, y, width, label, facecolor, edgecolor in path_stages:
        rounded_box(
            x,
            y,
            width,
            0.060,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.55,
        )
        axis.text(
            x + width / 2,
            y + 0.030,
            label,
            ha="center",
            va="center",
            fontsize=3.8,
            weight="bold",
            color=edgecolor,
        )
    for source_x, target_x in ((0.792, 0.797), (0.856, 0.861), (0.920, 0.925)):
        arrow((source_x, 0.880), (target_x, 0.880), width=0.6)

    axis.text(
        0.735,
        0.805,
        "Distinct frozen analyses; estimates are not pooled",
        fontsize=4.4,
        color=colors["neutral"],
    )

    evidence_rows = (
        (
            0.575,
            "1",
            "SUPPORT",
            "Trusted namespace at k=20",
            f"recall {recall_delta:+.1f} pp",
            f"adm. risk {risk_delta:+.1f} pp",
            "FIG. 2",
            colors["allowed"],
            colors["pale_green"],
        ),
        (
            0.335,
            "2",
            "VERIFICATION",
            "Stable-control overflip",
            f"{min(overflip_values):.1f}-{max(overflip_values):.1f}%",
            "across three text models",
            "FIG. 3",
            colors["trusted"],
            colors["pale_blue"],
        ),
        (
            0.095,
            "3",
            "EXPOSURE",
            "Relevant-inadmissible effect",
            f"{exposure_delta:+.1f} pp disclosure",
            "DeepSeek; 95% CI > 0",
            "FIG. 4",
            colors["blocked"],
            colors["pale_red"],
        ),
    )
    for y, number, heading, context, value, detail, figure_label, color, facecolor in evidence_rows:
        rounded_box(
            0.735,
            y,
            0.251,
            0.185,
            facecolor=facecolor,
            edgecolor="#FFFFFF",
            linewidth=0,
            radius=0.004,
        )
        axis.add_patch(Circle((0.753, y + 0.148), 0.011, facecolor=color, edgecolor="none"))
        axis.text(
            0.753,
            y + 0.148,
            number,
            ha="center",
            va="center",
            fontsize=4.8,
            weight="bold",
            color="#FFFFFF",
        )
        axis.text(
            0.771,
            y + 0.153,
            heading,
            fontsize=5.2,
            weight="bold",
            color=color,
            va="center",
        )
        axis.text(
            0.973,
            y + 0.153,
            figure_label,
            fontsize=4.4,
            weight="bold",
            color=colors["trusted"],
            ha="right",
            va="center",
        )
        axis.text(
            0.753,
            y + 0.105,
            context,
            fontsize=4.9,
            color=colors["neutral"],
        )
        axis.text(
            0.753,
            y + 0.062,
            value,
            fontsize=6.4,
            weight="bold",
            color=color,
        )
        axis.text(
            0.753,
            y + 0.027,
            detail,
            fontsize=5.2 if heading == "SUPPORT" else 4.6,
            weight="bold" if heading == "SUPPORT" else "normal",
            color=color if heading == "SUPPORT" else colors["neutral"],
        )

    axis.text(
        0.014,
        0.012,
        "Abridged audited RHELM/MemOps examples; examples establish failure types, not prevalence.",
        fontsize=3.8,
        color="#89919A",
    )

    figure.tight_layout(pad=0.12)
    _save_figure(figure, path)
    plt.close(figure)


def _write_pipeline_figure(
    path: Path,
    cases: Sequence[FigureCase],
    selected: Mapping[str, Row],
) -> None:
    """Render audited examples and the stage-localized evaluation framework."""
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    del selected

    figure, axis = plt.subplots(figsize=(7.2, 3.65))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    colors = {
        "ink": "#20262D",
        "muted": "#5E6872",
        "line": "#C9D0D7",
        "scope": "#2166A5",
        "lifecycle": "#B86A16",
        "policy": "#7A5195",
        "allow": "#248553",
        "deny": "#C43D3D",
        "pale_blue": "#EEF5FA",
        "pale_gold": "#FFF6E7",
        "pale_purple": "#F4EFF8",
        "pale_green": "#EAF5EE",
        "pale_red": "#FBEDEE",
        "pale_gray": "#F4F6F8",
    }

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        facecolor: str,
        edgecolor: str = "#C9D0D7",
        linewidth: float = 0.7,
        radius: float = 0.006,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0.004,rounding_size={radius}",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        )

    def arrow(x_start: float, y_start: float, x_end: float, y_end: float) -> None:
        axis.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(x_start, y_start),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colors["muted"],
                "linewidth": 0.8,
                "mutation_scale": 7,
                "shrinkA": 1,
                "shrinkB": 1,
            },
        )

    case_by_id = {str(case["case_id"]): case for case in cases}
    scope_case = case_by_id["wrong_namespace_market"]
    lifecycle_case = case_by_id["superseded_lisbon_date"]
    policy_case = case_by_id["forgotten_phone_detail"]
    history_case = case_by_id["historical_metformin_timeline"]
    scope_memory = next(
        memory for memory in scope_case["memories"] if memory["scope"] == "disallowed"
    )
    lifecycle_memory = next(
        memory for memory in lifecycle_case["memories"] if memory["state"] == "superseded"
    )
    policy_memory = next(
        memory
        for memory in policy_case["memories"]
        if memory["prohibited"] and not memory["usable"]
    )
    history_memory = next(
        memory for memory in history_case["memories"] if memory["state"] == "superseded"
    )

    axis.text(0.015, 0.985, "a", fontsize=9.5, weight="bold", va="top")
    axis.text(
        0.043,
        0.985,
        "Relevant candidates can still be inadmissible",
        fontsize=9.0,
        weight="bold",
        va="top",
        color=colors["ink"],
    )
    axis.text(
        0.043,
        0.935,
        "Abridged audited examples; verdicts are established by released fields",
        fontsize=7.0,
        color=colors["muted"],
        va="top",
    )

    columns = (
        (0.020, 0.165, "QUERY + INTENT"),
        (0.190, 0.225, "RETRIEVED CANDIDATE"),
        (0.420, 0.100, "CHECK"),
        (0.525, 0.070, "VERDICT"),
    )
    for x, width, label in columns:
        box(x, 0.850, width, 0.052, facecolor="#E9EDF1", edgecolor="#E9EDF1")
        axis.text(
            x + width / 2,
            0.876,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            weight="bold",
            color=colors["muted"],
        )

    example_rows = (
        (
            0.650,
            "RHELM | current answer",
            str(scope_case["query"]),
            str(scope_memory["text"]),
            "SCOPE",
            "wrong principal",
            "DROP",
            colors["scope"],
            colors["pale_blue"],
            colors["deny"],
        ),
        (
            0.455,
            "MemOps | current state",
            str(lifecycle_case["query"]),
            str(lifecycle_memory["text"]),
            "LIFECYCLE",
            "superseded now",
            "DROP",
            colors["lifecycle"],
            colors["pale_gold"],
            colors["deny"],
        ),
        (
            0.260,
            "MemOps | active policy",
            str(policy_case["query"]),
            str(policy_memory["text"]),
            "POLICY",
            "explicitly forgotten",
            "DROP",
            colors["policy"],
            colors["pale_purple"],
            colors["deny"],
        ),
        (
            0.065,
            "MemOps | history",
            str(history_case["query"]),
            str(history_memory["text"]),
            "LIFECYCLE",
            "old state required",
            "ADMIT",
            colors["lifecycle"],
            colors["pale_green"],
            colors["allow"],
        ),
    )
    for (
        y,
        intent,
        query,
        candidate,
        check,
        reason,
        decision,
        check_color,
        row_color,
        decision_color,
    ) in example_rows:
        box(0.020, y, 0.575, 0.175, facecolor=row_color)
        axis.plot([0.022, 0.022], [y + 0.010, y + 0.165], color=check_color, linewidth=3.0)
        axis.text(
            0.036,
            y + 0.143,
            intent,
            fontsize=7.0,
            weight="bold",
            color=check_color,
            va="center",
        )
        axis.text(
            0.036,
            y + 0.087,
            textwrap.fill(textwrap.shorten(query, width=62, placeholder="..."), width=22),
            fontsize=6.7,
            color=colors["ink"],
            va="center",
            linespacing=1.08,
        )
        axis.plot([0.185, 0.185], [y + 0.010, y + 0.165], color=colors["line"], linewidth=0.6)
        axis.text(
            0.200,
            y + 0.088,
            textwrap.fill(
                f'"{textwrap.shorten(candidate, width=64, placeholder="...")}"',
                width=25,
            ),
            fontsize=6.7,
            color=colors["ink"],
            va="center",
            linespacing=1.08,
        )
        axis.plot([0.415, 0.415], [y + 0.010, y + 0.165], color=colors["line"], linewidth=0.6)
        axis.text(
            0.470,
            y + 0.112,
            check,
            fontsize=6.8,
            weight="bold",
            color=check_color,
            ha="center",
        )
        axis.text(
            0.470,
            y + 0.057,
            textwrap.fill(reason, width=19),
            fontsize=6.2,
            color=colors["muted"],
            ha="center",
            va="center",
        )
        axis.plot([0.520, 0.520], [y + 0.010, y + 0.165], color=colors["line"], linewidth=0.6)
        axis.text(
            0.558,
            y + 0.088,
            decision,
            fontsize=7.2,
            weight="bold",
            color=decision_color,
            ha="center",
            va="center",
        )

    axis.plot([0.615, 0.615], [0.055, 0.965], color="#D7DCE1", linewidth=0.8)
    axis.text(0.635, 0.985, "b", fontsize=9.5, weight="bold", va="top")
    axis.text(
        0.663,
        0.985,
        "What the framework verifies",
        fontsize=8.8,
        weight="bold",
        va="top",
        color=colors["ink"],
    )
    axis.text(
        0.635,
        0.935,
        "Relevant candidates are checked before prompt assembly",
        fontsize=7.0,
        color=colors["muted"],
        va="top",
    )

    box(0.635, 0.790, 0.350, 0.100, facecolor=colors["pale_gray"])
    axis.text(
        0.810,
        0.852,
        "QUERY CONTEXT + TOPICAL CANDIDATE",
        fontsize=7.0,
        weight="bold",
        color=colors["ink"],
        ha="center",
        va="center",
    )
    axis.text(
        0.810,
        0.815,
        "query, intent, principal, time  |  memory m",
        fontsize=6.2,
        color=colors["muted"],
        ha="center",
        va="center",
    )

    arrow(0.810, 0.785, 0.810, 0.745)
    box(0.635, 0.585, 0.350, 0.155, facecolor="#FFFFFF")
    axis.text(
        0.650,
        0.712,
        "RECORD CHECK",
        fontsize=7.0,
        weight="bold",
        color=colors["ink"],
        va="center",
    )
    check_labels = (
        (0.670, "R", "relevant", colors["allow"], colors["pale_green"]),
        (0.750, "N", "scope", colors["scope"], colors["pale_blue"]),
        (0.830, "P", "policy", colors["policy"], colors["pale_purple"]),
        (0.910, "L", "lifecycle", colors["lifecycle"], colors["pale_gold"]),
    )
    for x, symbol, label, color, facecolor in check_labels:
        box(x - 0.029, 0.625, 0.058, 0.060, facecolor=facecolor, edgecolor=color)
        axis.text(
            x,
            0.661,
            symbol,
            fontsize=7.3,
            weight="bold",
            color=color,
            ha="center",
            va="center",
        )
        axis.text(
            x,
            0.637,
            label,
            fontsize=5.4,
            color=colors["muted"],
            ha="center",
            va="center",
        )
    axis.text(
        0.810,
        0.598,
        r"labels $\in\{1,0,?\}$; usable iff all four are 1",
        fontsize=6.2,
        color=colors["ink"],
        ha="center",
        va="center",
    )

    arrow(0.810, 0.580, 0.810, 0.540)
    box(0.635, 0.440, 0.350, 0.095, facecolor=colors["pale_green"], edgecolor="#AFC8B9")
    axis.text(
        0.810,
        0.503,
        "PRE-PROMPT DECISION",
        fontsize=7.0,
        weight="bold",
        color=colors["ink"],
        ha="center",
        va="center",
    )
    axis.text(
        0.810,
        0.466,
        "ADMIT  |  EXCLUDE + reason  |  UNRESOLVED",
        fontsize=6.3,
        color=colors["muted"],
        ha="center",
        va="center",
    )

    arrow(0.770, 0.435, 0.727, 0.395)
    arrow(0.850, 0.435, 0.893, 0.395)
    box(0.635, 0.290, 0.175, 0.100, facecolor=colors["pale_blue"])
    axis.text(
        0.723, 0.362, "ROUTE LEVEL", fontsize=6.8, weight="bold", color=colors["scope"], ha="center"
    )
    axis.text(
        0.723, 0.329, r"min prefix at $\tau=.8$", fontsize=6.1, color=colors["ink"], ha="center"
    )
    axis.text(
        0.723,
        0.302,
        r"recall, feasibility, $\mathrm{PCR}^{A}$",
        fontsize=5.9,
        color=colors["muted"],
        ha="center",
    )

    box(0.820, 0.290, 0.165, 0.100, facecolor=colors["pale_purple"])
    axis.text(
        0.903,
        0.362,
        "READER BOUNDARY",
        fontsize=6.8,
        weight="bold",
        color=colors["policy"],
        ha="center",
    )
    axis.text(0.903, 0.329, "retrieved -> exposed", fontsize=6.1, color=colors["ink"], ha="center")
    axis.text(0.903, 0.302, "-> disclosed", fontsize=6.1, color=colors["muted"], ha="center")

    axis.text(
        0.810,
        0.245,
        "NATURAL CLOSURE + CONTROLLED INTERVENTION\n(reader estimates never pooled)",
        fontsize=5.9,
        weight="bold",
        color=colors["muted"],
        ha="center",
        va="center",
        linespacing=1.05,
    )
    evidence_cards = (
        (0.635, "SUPPORT", "87 groups\n3,767 queries", colors["scope"], colors["pale_blue"]),
        (0.755, "END-TO-END", "1,523 cases\n3 readers", colors["policy"], colors["pale_purple"]),
        (0.875, "EXPOSURE", "16 scenarios\n4 readers", colors["deny"], colors["pale_red"]),
    )
    for x, heading, population, color, facecolor in evidence_cards:
        box(x, 0.065, 0.110, 0.145, facecolor=facecolor, edgecolor=color)
        axis.text(
            x + 0.055,
            0.173,
            heading,
            fontsize=6.4,
            weight="bold",
            color=color,
            ha="center",
            va="center",
        )
        axis.text(
            x + 0.055,
            0.112,
            population,
            fontsize=5.9,
            color=colors["ink"],
            ha="center",
            va="center",
            linespacing=1.15,
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
        ["Evidence recall", "Feasible rate", "V1 non-usable risk"],
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
    axes[1, 0].set_xlabel("Non-usable fraction within feasible matched prefixes")
    axes[1, 0].set_title(
        "(c) Unresolved labels keep the upper bound high", loc="left", weight="bold"
    )
    axes[1, 0].grid(False)
    effects = (
        ("Threshold recall", selected["threshold_recall_delta"], False, threshold_color),
        (
            "Threshold non-usable",
            selected["threshold_contamination_delta"],
            True,
            threshold_color,
        ),
        ("Cluster recall", selected["cluster_recall_delta"], False, cluster_color),
        (
            "Cluster non-usable",
            selected["cluster_contamination_delta"],
            True,
            cluster_color,
        ),
        (
            "Released-field v1 recall",
            selected["lifecycle_recall_delta"],
            False,
            lifecycle_color,
        ),
        (
            "Released-field v1 non-usable",
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


def _interval_values(
    row: Mapping[str, str], *, reverse: bool = False
) -> tuple[float, float, float]:
    estimate = _number(row)
    lower = _number(row, "ci95_lower")
    upper = _number(row, "ci95_upper")
    if reverse:
        return -estimate, -upper, -lower
    return estimate, lower, upper


def _write_constraint_reliability_figure(path: Path, selected: Mapping[str, Row]) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
        }
    )
    figure = plt.figure(figsize=(7.1, 3.6), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.16, 1.0, 1.40),
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.18,
        wspace=0.58,
    )
    frontier_axis = figure.add_subplot(grid[0, 0])
    attribution_axis = figure.add_subplot(grid[0, 1])
    bracket_axis = figure.add_subplot(grid[0, 2])

    colors = {
        "global": "#555D66",
        "namespace": "#087F8C",
        "policy": "#2E8B57",
        "lifecycle": "#B64342",
        "combined": "#2F6B9A",
        "dominant": "#2E8B57",
        "neutral": "#5B6168",
        "interval": "#D89B45",
        "bad": "#B64342",
    }

    for top_k in (10, 20, 50, 100):
        global_risk = _number(selected[f"c9_k{top_k}_global_penalized_admissibility_upper_risk"])
        namespace_risk = _number(
            selected[f"c9_k{top_k}_namespace_penalized_admissibility_upper_risk"]
        )
        global_recall = _number(selected[f"c9_k{top_k}_global_evidence_recall"])
        namespace_recall = _number(selected[f"c9_k{top_k}_namespace_evidence_recall"])
        frontier_axis.add_patch(
            FancyArrowPatch(
                (global_risk, global_recall),
                (namespace_risk, namespace_recall),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.4,
                color=colors["namespace"],
                alpha=0.9,
                zorder=2,
            )
        )
        frontier_axis.scatter(
            global_risk,
            global_recall,
            s=24,
            facecolor="#FFFFFF",
            edgecolor=colors["global"],
            linewidth=1.0,
            zorder=3,
        )
        frontier_axis.scatter(
            namespace_risk,
            namespace_recall,
            s=28,
            facecolor=colors["namespace"],
            edgecolor="#FFFFFF",
            linewidth=0.7,
            zorder=4,
        )
        frontier_axis.annotate(
            f"k={top_k}",
            (namespace_risk, namespace_recall),
            xytext=(4, -11 if top_k in {50, 100} else 3),
            textcoords="offset points",
            fontsize=7.0,
            weight="bold",
            color=colors["namespace"],
        )
        if top_k in {20, 100}:
            feasible_delta = _number(selected[f"c9_k{top_k}_feasible_rate_delta"])
            frontier_axis.text(
                (global_risk + namespace_risk) / 2,
                (global_recall + namespace_recall) / 2 - (0.030 if top_k == 20 else 0.040),
                rf"$\Delta F={feasible_delta:+.3f}$",
                fontsize=6.5,
                ha="center",
                color=colors["neutral"],
            )
    frontier_axis.set_xlim(0.30, 0.88)
    frontier_axis.set_ylim(0.27, 0.92)
    frontier_axis.set_xlabel(r"Penalized admissibility risk  $\leftarrow$ better")
    frontier_axis.set_ylabel("Evidence recall  (higher is better)")
    frontier_axis.set_title(
        "a  Constraints improve the frontier",
        loc="left",
        weight="bold",
        fontsize=9.2,
    )
    frontier_axis.grid(color="#E7E9EC", linewidth=0.55)
    frontier_axis.plot(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="#FFFFFF",
        markeredgecolor=colors["global"],
        label="Global",
    )
    frontier_axis.plot(
        [],
        [],
        marker="o",
        linestyle="none",
        color=colors["namespace"],
        label="Namespace",
    )
    frontier_axis.legend(loc="lower left", fontsize=7.0, handletextpad=0.3)

    methods = (
        ("policy_only", "Policy only", colors["policy"], (-55, 7)),
        ("lifecycle_only", "Lifecycle only", colors["lifecycle"], (5, -13)),
        ("governance_v2", "Policy + lifecycle", colors["combined"], (-47, -2)),
    )
    attribution_axis.add_patch(
        Rectangle((0, 0), 0.18, 0.05, facecolor="#EAF5EE", edgecolor="none", zorder=-4)
    )
    attribution_axis.axvline(0, color="#7D848B", linewidth=0.75)
    attribution_axis.axhline(0, color="#7D848B", linewidth=0.75)
    for method, label, color, offset in methods:
        x, x_low, x_high = _interval_values(
            selected[f"c10_{method}_penalized_admissibility_upper_risk_delta"], reverse=True
        )
        y, y_low, y_high = _interval_values(selected[f"c10_{method}_evidence_recall_delta"])
        feasible = _number(selected[f"c10_{method}_feasible_rate_delta"])
        attribution_axis.errorbar(
            x,
            y,
            xerr=[[x - x_low], [x_high - x]],
            yerr=[[y - y_low], [y_high - y]],
            fmt="o",
            color=color,
            markersize=5.0,
            capsize=2.0,
            linewidth=1.0,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.5,
            zorder=3,
        )
        attribution_axis.annotate(
            f"{label}\n" + rf"$\Delta F={feasible:+.3f}$",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.6,
            color=color,
            weight="bold",
            linespacing=1.05,
        )
    attribution_axis.set_xlim(-0.035, 0.175)
    attribution_axis.set_ylim(-0.025, 0.040)
    attribution_axis.set_xlabel("Risk reduction  (right is better)")
    attribution_axis.set_ylabel("Recall gain  (up is better)")
    attribution_axis.set_title(
        "b  Policy creates the gain", loc="left", weight="bold", fontsize=9.2
    )
    attribution_axis.grid(color="#E7E9EC", linewidth=0.55)

    channels = (
        ("namespace_false_deny", "False deny"),
        ("namespace_missing", "Missing label"),
        ("namespace_swap", "Source-label swap"),
        ("namespace_false_allow", "False allow"),
        ("policy_false_deny", "Policy false deny"),
        ("lifecycle_false_stale", "Lifecycle false stale"),
        ("policy_false_allow", "Policy false allow"),
        ("policy_missing", "Policy missing / fail open"),
    )
    y_positions = np.array((0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0))
    bracket_axis.axhspan(-0.45, 3.45, color="#EEF4FA", zorder=-4)
    bracket_axis.axhspan(4.55, 8.45, color="#FFF6E7", zorder=-4)
    for y_position, (channel, _label) in zip(y_positions, channels, strict=True):
        last = _number(selected[f"c10_{channel}_last"])
        bracket_axis.plot(
            [0, last], [y_position, y_position], color=colors["dominant"], linewidth=3
        )
        first_key = f"c10_{channel}_first"
        if first_key in selected:
            first = _number(selected[first_key])
            bracket_axis.plot(
                [last, first],
                [y_position, y_position],
                color=colors["interval"],
                linewidth=3,
            )
            bracket_axis.scatter([first], [y_position], marker="x", color=colors["bad"], s=22)
        else:
            bracket_axis.annotate(
                "",
                xy=(0.525, y_position),
                xytext=(last, y_position),
                arrowprops={"arrowstyle": "->", "color": colors["dominant"], "lw": 1.2},
            )
    bracket_axis.set_yticks(y_positions, [label for _channel, label in channels])
    bracket_axis.set_ylim(8.65, -1.05)
    bracket_axis.set_xlim(0, 0.55)
    bracket_axis.set_xticks(np.arange(0, 0.6, 0.1))
    bracket_axis.set_xlabel("Metadata corruption rate")
    bracket_axis.set_title(
        "c  Reliability is channel-specific", loc="left", weight="bold", fontsize=9.2
    )
    bracket_axis.grid(axis="x", color="#E7E9EC", linewidth=0.6)
    bracket_axis.text(
        0.0,
        -0.70,
        "NAMESPACE  |  reference: global dense",
        fontsize=7.0,
        weight="bold",
        color=colors["combined"],
    )
    bracket_axis.text(
        0.0,
        4.30,
        "GOVERNANCE  |  reference: clean namespace dense",
        fontsize=7.0,
        weight="bold",
        color=colors["interval"],
    )
    figure.text(0.20, 0.965, "VALUE", ha="center", fontsize=8.0, weight="bold", color="#6B7279")
    figure.text(0.50, 0.965, "SOURCE", ha="center", fontsize=8.0, weight="bold", color="#6B7279")
    figure.text(0.82, 0.965, "FRAGILITY", ha="center", fontsize=8.0, weight="bold", color="#6B7279")
    figure.text(0.35, 0.965, r"$\longrightarrow$", ha="center", fontsize=9.0, color="#A0A6AC")
    figure.text(0.66, 0.965, r"$\longrightarrow$", ha="center", fontsize=9.0, color="#A0A6AC")

    _save_figure(figure, path)
    plt.close(figure)


def _write_inference_gap_figure(path: Path, selected: Mapping[str, Row]) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.2),
        gridspec_kw={"width_ratios": (1.0, 1.12)},
        layout="constrained",
    )
    colors = {
        "oracle": "#2E8B57",
        "openai": "#195B9A",
        "gemini": "#D9822B",
        "deepseek": "#7B5AA6",
        "risk": "#B64342",
    }

    contrasts = (
        ("released_oracle", "Released oracle", colors["oracle"]),
        ("openai_text_inferred", "GPT-5.6", colors["openai"]),
        ("gemini_text_inferred", "Gemini 3.6", colors["gemini"]),
    )
    axes[0].add_patch(
        Rectangle((0, 0), 0.03, 0.08, facecolor="#EAF5EE", edgecolor="none", zorder=-5)
    )
    axes[0].add_patch(
        Rectangle((-0.12, -0.08), 0.12, 0.08, facecolor="#FCEEEE", edgecolor="none", zorder=-5)
    )
    axes[0].add_patch(
        Rectangle((-0.12, 0), 0.12, 0.08, facecolor="#FFF6E8", edgecolor="none", zorder=-5)
    )
    axes[0].axvline(0, color="#737980", linewidth=0.8)
    axes[0].axhline(0, color="#737980", linewidth=0.8)
    label_offsets = {
        "released_oracle": (7, 6),
        "openai_text_inferred": (6, -15),
        "gemini_text_inferred": (7, -15),
    }
    for contrast, display, color in contrasts:
        x, x_lower, x_upper = _interval_values(selected[f"c11_{contrast}_feasible_rate_delta"])
        y, y_lower, y_upper = _interval_values(
            selected[f"c11_{contrast}_penalized_admissibility_upper_risk_delta"], reverse=True
        )
        recall = _number(selected[f"c11_{contrast}_evidence_recall_delta"])
        axes[0].errorbar(
            x,
            y,
            xerr=[[x - x_lower], [x_upper - x]],
            yerr=[[y - y_lower], [y_upper - y]],
            fmt="o",
            color=color,
            markersize=5.5,
            capsize=2.2,
            linewidth=1.1,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.6,
            zorder=3,
        )
        axes[0].annotate(
            f"{display}\n" + rf"$\Delta R={recall:+.3f}$",
            (x, y),
            xytext=label_offsets[contrast],
            textcoords="offset points",
            fontsize=6.8,
            weight="bold",
            color=color,
            linespacing=1.05,
        )
    axes[0].text(0.014, 0.067, "IDEAL", fontsize=6.5, weight="bold", color=colors["oracle"])
    axes[0].text(
        -0.115,
        0.067,
        "safer, but\nover-refuses",
        fontsize=6.0,
        color="#9A6D2F",
        va="top",
    )
    axes[0].text(-0.115, -0.073, "worse on both", fontsize=6.0, color=colors["risk"])
    axes[0].set_xlim(-0.12, 0.03)
    axes[0].set_ylim(-0.08, 0.08)
    axes[0].set_xlabel(r"$\Delta$ feasible rate  (right is better)")
    axes[0].set_ylabel("Admissibility-risk reduction  (up is better)")
    axes[0].set_title("a  Released fields help; text inference does not", loc="left", weight="bold")
    axes[0].grid(color="#E7E9EC", linewidth=0.55)

    controlled = (
        ("gpt56_controlled", "GPT-5.6", colors["openai"]),
        ("gemini_controlled", "Gemini 3.6", colors["gemini"]),
        ("deepseek_controlled", "DeepSeek-V4", colors["deepseek"]),
    )
    axes[1].add_patch(
        Rectangle((0, 0.8), 0.05, 0.22, facecolor="#DCEEDB", edgecolor="none", zorder=0)
    )
    for contrast, label, color in controlled:
        x_row = selected[f"c11_{contrast}_stable_control_overflip_rate"]
        y_row = selected[f"c11_{contrast}_strict_focal_pair_consistency"]
        x, x_lower, x_upper = _interval_values(x_row)
        y, y_lower, y_upper = _interval_values(y_row)
        false_deny = _number(selected[f"c11_{contrast}_stable_admissible_false_deny_rate"])
        axes[1].errorbar(
            x,
            y,
            xerr=[[x - x_lower], [x_upper - x]],
            yerr=[[y - y_lower], [y_upper - y]],
            fmt="o",
            color=color,
            markersize=5.8,
            capsize=2,
            linewidth=1.1,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.6,
        )
        short_label = {
            "GPT-5.6": "GPT",
            "Gemini 3.6": "Gemini",
            "DeepSeek-V4": "DeepSeek",
        }[label]
        label_offset = (6, 6) if short_label == "DeepSeek" else (4, -16)
        axes[1].annotate(
            f"{short_label}\nFD={false_deny:.3f}",
            (x, y),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=7.0,
            color=color,
            weight="bold",
            va="bottom" if short_label == "DeepSeek" else "top",
        )
    axes[1].axvline(0.05, color="#6E8B6A", linestyle="--", linewidth=0.7)
    axes[1].axhline(0.8, color="#6E8B6A", linestyle="--", linewidth=0.7)
    axes[1].text(
        0.012,
        1.005,
        "supportive region",
        fontsize=6.2,
        color="#547651",
        va="top",
    )
    axes[1].set_xlim(-0.01, 0.41)
    axes[1].set_ylim(0.5, 1.03)
    axes[1].set_xlabel("Stable overflip  (lower is better)")
    axes[1].set_ylabel("Focal consistency  (higher is better)")
    axes[1].set_title(
        "b  Correct focal flips do not imply stable decisions", loc="left", weight="bold"
    )
    axes[1].grid(color="#E7E9EC", linewidth=0.6)

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
    natural_end_to_end_table_path = output_root / "natural_end_to_end_results.tex"
    outputs = (
        numbers_path,
        main_table_path,
        full_arm_table_path,
        matched_prefix_table_path,
        smoke_table_path,
        natural_end_to_end_table_path,
        output_root / "verification_pipeline.pdf",
        output_root / "verification_pipeline.png",
        output_root / "verification_pipeline.svg",
        output_root / "verification_examples.pdf",
        output_root / "verification_examples.png",
        output_root / "verification_examples.svg",
        output_root / "evidence_summary.pdf",
        output_root / "evidence_summary.png",
        output_root / "evidence_summary.svg",
        output_root / "retrieval_results.pdf",
        output_root / "retrieval_results.png",
        output_root / "retrieval_results.svg",
        output_root / "constraint_reliability.pdf",
        output_root / "constraint_reliability.png",
        output_root / "constraint_reliability.svg",
        output_root / "inference_gap.pdf",
        output_root / "inference_gap.png",
        output_root / "inference_gap.svg",
    )
    _write_numbers(numbers_path, selected)
    _write_main_table(main_table_path, selected)
    _write_full_arm_table(full_arm_table_path, rows)
    _write_matched_prefix_table(matched_prefix_table_path, selected)
    _write_smoke_table(smoke_table_path, selected)
    _write_natural_end_to_end_table(natural_end_to_end_table_path, selected)
    _write_pipeline_figure(output_root / "verification_pipeline", figure_examples, selected)
    _write_verification_examples_figure(output_root / "verification_examples", figure_examples)
    _write_evidence_figure(output_root / "evidence_summary", selected)
    _write_retrieval_figure(output_root / "retrieval_results", rows, selected)
    _write_constraint_reliability_figure(output_root / "constraint_reliability", selected)
    _write_inference_gap_figure(output_root / "inference_gap", selected)
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
