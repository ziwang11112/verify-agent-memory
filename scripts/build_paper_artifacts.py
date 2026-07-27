"""Generate paper numbers, tables, and figures from verified normalized evidence."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.verify_evidence import validate_evidence
from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

Row = dict[str, str]


def _load_rows(repository_root: Path) -> list[Row]:
    errors = validate_evidence(repository_root)
    if errors:
        raise ValueError("evidence verification failed: " + "; ".join(errors))
    rows: list[Row] = []
    for path in sorted((repository_root / "evidence" / "normalized").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


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


def _ci(row: Mapping[str, str], *, signed: bool = True, digits: int = 3) -> str:
    formatter = _signed if signed else _plain
    return (
        f"[{formatter(_number(row, 'ci95_lower'), digits)}, "
        f"{formatter(_number(row, 'ci95_upper'), digits)}]"
    )


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{\ensuremath{{{value}}}}}"


def _evidence_rows(rows: Sequence[Row]) -> dict[str, Row]:
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
        "prohibited_exact": _one(
            rows,
            claim_id="C8",
            contrast="prohibited",
            metric="exact_agreement",
        ),
        "prohibited_alpha": _one(
            rows,
            claim_id="C8",
            contrast="prohibited",
            metric="krippendorff_alpha",
        ),
    }


def _write_numbers(path: Path, selected: Mapping[str, Row]) -> None:
    macros = [
        "% Generated by scripts/build_paper_artifacts.py. Do not edit.",
        _macro("NaturalGroupCount", "87"),
        _macro("NaturalMemoryCount", "182{,}908"),
        _macro("NaturalQueryCount", "3{,}767"),
        _macro("NaturalRouteRows", "33{,}903"),
        _macro("HumanRecordCount", "207"),
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
        _macro("GOneLeakageDelta", _signed(_number(selected["g1_leakage"]))),
        _macro("GOneLeakageDeltaCI", _ci(selected["g1_leakage"])),
        _macro("GOneUtilityDelta", _signed(_number(selected["g1_utility"]))),
        _macro("GOneRefusalDelta", _signed(_number(selected["g1_refusal"]))),
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
            "ClusterRecallDelta",
            _signed(_number(selected["cluster_recall_delta"]), digits=4),
        ),
        _macro(
            "ClusterContaminationDelta",
            _signed(_number(selected["cluster_contamination_delta"]), digits=4),
        ),
        _macro(
            "LifecycleRecallDelta",
            _signed(_number(selected["lifecycle_recall_delta"])),
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
            "LifecycleStaleDelta",
            _signed(_number(selected["lifecycle_stale_delta"])),
        ),
        _macro(
            "LifecycleStaleReduction",
            _plain(abs(_number(selected["lifecycle_stale_delta"]))),
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
            "ProhibitedExactAgreement",
            _plain(_number(selected["prohibited_exact"])),
        ),
        _macro(
            "ProhibitedAlpha",
            _plain(_number(selected["prohibited_alpha"])),
        ),
    ]
    path.write_text("\n".join(macros) + "\n", encoding="ascii")


def _write_main_table(path: Path, selected: Mapping[str, Row]) -> None:
    entries = [
        (
            "Exposure/use",
            "Reader A risk difference",
            _plain(_number(selected["reader_a"])),
            _ci(selected["reader_a"], signed=False),
            "C2",
        ),
        (
            "Exposure/use",
            "Reader B risk difference",
            _plain(_number(selected["reader_b"])),
            _ci(selected["reader_b"], signed=False),
            "C2",
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
            "Conservative contamination",
            _signed(_number(selected["namespace_contamination_delta"])),
            _ci(selected["namespace_contamination_delta"]),
            "C5",
        ),
        (
            "Threshold vs. namespace",
            "Evidence recall",
            _signed(_number(selected["threshold_recall_delta"])),
            _ci(selected["threshold_recall_delta"]),
            "C6",
        ),
        (
            "Cluster vs. namespace",
            "Evidence recall",
            _signed(_number(selected["cluster_recall_delta"]), digits=4),
            _ci(selected["cluster_recall_delta"], digits=4),
            "C6",
        ),
        (
            "Lifecycle upper bound",
            "Conservative contamination",
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
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_agreement_table(path: Path, rows: Sequence[Row]) -> None:
    labels = {
        "relevance": "Relevance",
        "scope": "Scope",
        "state": "Lifecycle state",
        "prohibited": "Prohibited evidence",
        "usable_evidence": "Usable evidence",
    }
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Axis & Exact agreement & Krippendorff's $\alpha$ \\",
        r"\midrule",
    ]
    for axis, label in labels.items():
        exact = _one(rows, claim_id="C8", contrast=axis, metric="exact_agreement")
        alpha = _one(
            rows,
            claim_id="C8",
            contrast=axis,
            metric="krippendorff_alpha",
        )
        lines.append(f"{label} & {_plain(_number(exact))} & {_plain(_number(alpha))} \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


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
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
    figure.savefig(
        stem.with_suffix(".png"),
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "verify-agent-memory"},
    )


def _write_pipeline_figure(path: Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(7.0, 1.65))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = (
        ("Stored", "persistence", "#4C78A8"),
        ("Retrieved", "candidate selection", "#59A14F"),
        ("Exposed", "prompt assembly", "#F28E2B"),
        ("Used", "answer evidence", "#B55A30"),
    )
    centers = (0.11, 0.37, 0.63, 0.89)
    for index, ((title, subtitle, color), center) in enumerate(zip(stages, centers, strict=True)):
        box = FancyBboxPatch(
            (center - 0.09, 0.43),
            0.18,
            0.33,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        axis.add_patch(box)
        axis.text(center, 0.64, title, ha="center", va="center", weight="bold", color=color)
        axis.text(center, 0.51, subtitle, ha="center", va="center", fontsize=7, color="#333333")
        if index < len(stages) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (center + 0.095, 0.595),
                    (centers[index + 1] - 0.095, 0.595),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.0,
                    color="#555555",
                )
            )
    axis.text(
        0.5,
        0.24,
        "admissible = scope allowed  +  policy allowed  +  lifecycle compatible",
        ha="center",
        va="center",
        weight="bold",
        color="#222222",
    )
    axis.text(
        0.5,
        0.10,
        "usable = relevant + admissible     |     unknown labels remain unresolved",
        ha="center",
        va="center",
        color="#555555",
    )
    figure.tight_layout(pad=0.2)
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


def _write_evidence_figure(path: Path, rows: Sequence[Row], selected: Mapping[str, Row]) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

    _errorbar(
        axes[0],
        [selected["reader_a"], selected["reader_b"]],
        ["Reader A", "Reader B"],
        ["#007C91", "#D55E00"],
    )
    axes[0].set_title("(a) Exposure and answer use")
    axes[0].set_xlabel("Risk difference")
    axes[0].set_xlim(0.35, 0.60)

    retrieval_rows = [
        selected["namespace_recall_delta"],
        selected["namespace_contamination_delta"],
        selected["threshold_recall_delta"],
        selected["cluster_recall_delta"],
        selected["lifecycle_contamination_delta"],
    ]
    _errorbar(
        axes[1],
        retrieval_rows,
        [
            "Namespace: recall",
            "Namespace: contamination",
            "Threshold: recall",
            "Cluster: recall",
            "Lifecycle UB: contamination",
        ],
        ["#59A14F", "#59A14F", "#E15759", "#4C78A8", "#F28E2B"],
    )
    axes[1].set_title("(b) Retrieval deltas")
    axes[1].set_xlabel("Method minus reference")
    axes[1].set_xlim(-0.055, 0.18)

    axes_order = ("relevance", "scope", "state", "prohibited", "usable_evidence")
    labels = ("Relevance", "Scope", "Lifecycle", "Prohibited", "Usable")
    exact = [
        _number(_one(rows, claim_id="C8", contrast=axis, metric="exact_agreement"))
        for axis in axes_order
    ]
    alpha = [
        _number(_one(rows, claim_id="C8", contrast=axis, metric="krippendorff_alpha"))
        for axis in axes_order
    ]
    positions = list(range(len(labels)))
    width = 0.36
    axes[2].barh(
        [position - width / 2 for position in positions],
        exact,
        height=width,
        label="Exact",
        color="#4C78A8",
    )
    axes[2].barh(
        [position + width / 2 for position in positions],
        alpha,
        height=width,
        label=r"$\alpha$",
        color="#F2CF5B",
    )
    axes[2].set_yticks(positions, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1.02)
    axes[2].set_xlabel("Agreement")
    axes[2].set_title("(c) Human audit")
    axes[2].grid(axis="x", color="#E5E5E5", linewidth=0.6)
    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )

    figure.tight_layout(w_pad=1.2)
    _save_figure(figure, path)
    plt.close(figure)


def _write_manifest(
    repository_root: Path,
    output_root: Path,
    outputs: Sequence[Path],
) -> None:
    evidence_inputs = sorted((repository_root / "evidence" / "normalized").glob("*.csv"))
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
    selected = _evidence_rows(rows)
    output_root = repository_root / "paper" / "generated"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / "paper_numbers.tex",
        output_root / "main_results.tex",
        output_root / "human_agreement.tex",
        output_root / "mechanism_smoke.tex",
        output_root / "verification_pipeline.pdf",
        output_root / "verification_pipeline.png",
        output_root / "evidence_summary.pdf",
        output_root / "evidence_summary.png",
    )
    _write_numbers(outputs[0], selected)
    _write_main_table(outputs[1], selected)
    _write_agreement_table(outputs[2], rows)
    _write_smoke_table(outputs[3], selected)
    _write_pipeline_figure(output_root / "verification_pipeline")
    _write_evidence_figure(output_root / "evidence_summary", rows, selected)
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
