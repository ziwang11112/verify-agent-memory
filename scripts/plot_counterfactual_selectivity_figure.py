"""Generate the publication figure for counterfactual verifier selectivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "counterfactual_admissibility"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures"

MODEL_ORDER = (
    "OpenAI/gpt-5.6-sol",
    "DeepSeek/deepseek-v4-pro",
    "Gemini/gemini-3.6-flash",
)
MODEL_LABELS = {
    "OpenAI/gpt-5.6-sol": "GPT-5.6",
    "DeepSeek/deepseek-v4-pro": "DeepSeek-V4",
    "Gemini/gemini-3.6-flash": "Gemini 3.6",
    "released_oracle": "Released oracle",
    "no_verifier_keep_all": "No verifier",
}
MODEL_COLORS = {
    "OpenAI/gpt-5.6-sol": "#0F4D92",
    "DeepSeek/deepseek-v4-pro": "#42949E",
    "Gemini/gemini-3.6-flash": "#9A4D8E",
}
AXIS_ORDER = ("principal_scope", "policy_purpose", "lifecycle_intent", "as_of_time")
AXIS_LABELS = {
    "principal_scope": "Scope",
    "policy_purpose": "Policy",
    "lifecycle_intent": "Lifecycle",
    "as_of_time": "As-of time",
}
FIGURE_BASENAME = "counterfactual_selectivity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("source-data rows must be nonempty")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("source-data rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _one(rows: Sequence[Mapping[str, str]], **conditions: str) -> Mapping[str, str]:
    matches = [
        row for row in rows if all(row.get(field) == value for field, value in conditions.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions!r}, found {len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class Interval:
    value: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0 <= self.lower <= self.value <= self.upper <= 1:
            raise ValueError("rate interval must be ordered within [0, 1]")


@dataclass(frozen=True)
class ModelFigureData:
    model: str
    focal: Interval
    overflip: Interval
    stable_false_deny: float
    stable_false_admit: float
    axis_overflip: Mapping[str, float]


@dataclass(frozen=True)
class FigureData:
    models: tuple[ModelFigureData, ...]
    oracle_focal: float
    oracle_overflip: float
    no_verifier_focal: float
    no_verifier_overflip: float


def _interval(
    aggregate: Sequence[Mapping[str, str]],
    bootstrap: Sequence[Mapping[str, str]],
    *,
    model: str,
    aggregate_field: str,
    bootstrap_metric: str,
) -> Interval:
    value = float(_one(aggregate, model=model, scope="overall")[aggregate_field])
    ci = _one(
        bootstrap,
        model=model,
        scope="overall",
        metric=bootstrap_metric,
    )
    return Interval(value=value, lower=float(ci["ci_lower"]), upper=float(ci["ci_upper"]))


def load_figure_data(results_dir: Path) -> FigureData:
    """Load only published aggregate and posthoc source tables."""
    aggregate = _read_csv(results_dir / "aggregate_metrics.csv")
    bootstrap = _read_csv(results_dir / "bootstrap_ci.csv")
    taxonomy = _read_csv(results_dir / "error_taxonomy.csv")
    stable = _read_csv(results_dir / "stable_control_outcomes.csv")
    models = []
    for model in MODEL_ORDER:
        false_deny = float(
            _one(
                taxonomy,
                model=model,
                axis="all",
                role="stable_admissible",
                condition="all",
            )["false_deny_rate"]
        )
        false_admit = float(
            _one(
                taxonomy,
                model=model,
                axis="all",
                role="stable_inadmissible",
                condition="all",
            )["false_admit_rate"]
        )
        axis_overflip = {
            axis: float(
                _one(
                    stable,
                    model=model,
                    axis=axis,
                    role="all_stable",
                )["overflip_rate"]
            )
            for axis in AXIS_ORDER
        }
        models.append(
            ModelFigureData(
                model=model,
                focal=_interval(
                    aggregate,
                    bootstrap,
                    model=model,
                    aggregate_field="strict_focal_pair_consistency",
                    bootstrap_metric="strict_focal_pair_consistency",
                ),
                overflip=_interval(
                    aggregate,
                    bootstrap,
                    model=model,
                    aggregate_field="stable_control_overflip_rate",
                    bootstrap_metric="stable_control_overflip_rate",
                ),
                stable_false_deny=false_deny,
                stable_false_admit=false_admit,
                axis_overflip=axis_overflip,
            )
        )
    oracle = _one(aggregate, model="released_oracle", scope="overall")
    no_verifier = _one(aggregate, model="no_verifier_keep_all", scope="overall")
    return FigureData(
        models=tuple(models),
        oracle_focal=float(oracle["strict_focal_pair_consistency"]),
        oracle_overflip=float(oracle["stable_control_overflip_rate"]),
        no_verifier_focal=float(no_verifier["strict_focal_pair_consistency"]),
        no_verifier_overflip=float(no_verifier["stable_control_overflip_rate"]),
    )


def source_data_rows(data: FigureData) -> tuple[dict[str, object], ...]:
    """Return tidy source rows for every plotted quantity."""
    rows = []
    for model in data.models:
        for metric, interval, sample_size in (
            ("strict_focal_pair_consistency", model.focal, 32),
            ("stable_control_overflip_rate", model.overflip, 64),
        ):
            rows.append(
                {
                    "panel": "a",
                    "model": model.model,
                    "display_name": MODEL_LABELS[model.model],
                    "axis": "all",
                    "metric": metric,
                    "value": interval.value,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "n": sample_size,
                    "analysis_status": "preregistered",
                }
            )
        for metric, value in (
            ("stable_admissible_false_deny_rate", model.stable_false_deny),
            ("stable_inadmissible_false_admit_rate", model.stable_false_admit),
        ):
            rows.append(
                {
                    "panel": "b",
                    "model": model.model,
                    "display_name": MODEL_LABELS[model.model],
                    "axis": "all",
                    "metric": metric,
                    "value": value,
                    "ci_lower": "",
                    "ci_upper": "",
                    "n": 64,
                    "analysis_status": "posthoc_descriptive",
                }
            )
        for axis in AXIS_ORDER:
            rows.append(
                {
                    "panel": "c",
                    "model": model.model,
                    "display_name": MODEL_LABELS[model.model],
                    "axis": axis,
                    "metric": "stable_control_overflip_rate",
                    "value": model.axis_overflip[axis],
                    "ci_lower": "",
                    "ci_upper": "",
                    "n": 16,
                    "analysis_status": "posthoc_descriptive",
                }
            )
    for model, focal, overflip in (
        ("released_oracle", data.oracle_focal, data.oracle_overflip),
        ("no_verifier_keep_all", data.no_verifier_focal, data.no_verifier_overflip),
    ):
        for metric, value, sample_size in (
            ("strict_focal_pair_consistency", focal, 32),
            ("stable_control_overflip_rate", overflip, 64),
        ):
            rows.append(
                {
                    "panel": "a",
                    "model": model,
                    "display_name": MODEL_LABELS[model],
                    "axis": "all",
                    "metric": metric,
                    "value": value,
                    "ci_lower": value,
                    "ci_upper": value,
                    "n": sample_size,
                    "analysis_status": "reference",
                }
            )
    return tuple(rows)


def apply_style() -> None:
    """Apply the exclusive Python publication style before creating a figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "svg.hashsalt": "counterfactual-selectivity-v1",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _plot_panel_a(ax: plt.Axes, data: FigureData) -> None:
    ax.add_patch(
        Rectangle(
            (-0.01, 0.8),
            0.06,
            0.23,
            facecolor="#DDF3DE",
            edgecolor="#2E7D32",
            linewidth=0.7,
            hatch="///",
            alpha=0.55,
            zorder=0,
        )
    )
    ax.axvline(0.05, color="#767676", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.8, color="#767676", lw=0.8, ls=(0, (3, 2)), zorder=1)
    for model in data.models:
        xerr = np.array(
            [
                [model.overflip.value - model.overflip.lower],
                [model.overflip.upper - model.overflip.value],
            ]
        )
        yerr = np.array(
            [[model.focal.value - model.focal.lower], [model.focal.upper - model.focal.value]]
        )
        ax.errorbar(
            model.overflip.value,
            model.focal.value,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=5.5,
            mfc=MODEL_COLORS[model.model],
            mec="white",
            mew=0.7,
            ecolor=MODEL_COLORS[model.model],
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
    ax.plot(
        data.oracle_overflip,
        data.oracle_focal,
        marker="*",
        ms=7,
        color="#272727",
        zorder=4,
    )
    ax.plot(
        data.no_verifier_overflip,
        data.no_verifier_focal,
        marker="D",
        ms=4.2,
        mfc="white",
        mec="#767676",
        mew=0.9,
        zorder=4,
    )
    offsets = {
        "OpenAI/gpt-5.6-sol": (5, -9),
        "DeepSeek/deepseek-v4-pro": (5, -1),
        "Gemini/gemini-3.6-flash": (5, 6),
    }
    for model in data.models:
        ax.annotate(
            MODEL_LABELS[model.model],
            (model.overflip.value, model.focal.value),
            xytext=offsets[model.model],
            textcoords="offset points",
            color=MODEL_COLORS[model.model],
            fontsize=6.2,
            fontweight="bold",
        )
    ax.annotate("Oracle", (0, 1), xytext=(5, -2), textcoords="offset points", fontsize=6)
    ax.annotate(
        "No verifier",
        (0, 0),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=6,
        color="#606060",
    )
    ax.text(0.006, 0.815, "supportive\nregion", fontsize=5.7, color="#2E7D32")
    ax.set_xlim(-0.015, 0.37)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks((0, 0.1, 0.2, 0.3))
    ax.set_yticks((0, 0.4, 0.8, 1.0))
    ax.set_xlabel("Stable-control overflip rate (lower is better)")
    ax.set_ylabel("Strict focal-pair consistency (higher is better)")
    ax.set_title("Focal success is not selectivity", loc="left", fontweight="bold")
    _panel_label(ax, "a")


def _plot_panel_b(ax: plt.Axes, data: FigureData) -> None:
    y = np.arange(len(data.models))[::-1]
    for yi, model in zip(y, data.models, strict=True):
        values = (model.stable_false_admit, model.stable_false_deny)
        ax.plot(values, (yi, yi), color="#CFCECE", lw=1.4, zorder=1)
        ax.plot(
            model.stable_false_deny,
            yi,
            marker="o",
            ms=5,
            color=MODEL_COLORS[model.model],
            zorder=3,
        )
        ax.plot(
            model.stable_false_admit,
            yi,
            marker="s",
            ms=4.5,
            mfc="white",
            mec=MODEL_COLORS[model.model],
            mew=1,
            zorder=3,
        )
        ax.text(
            model.stable_false_deny + 0.018,
            yi,
            f"{model.stable_false_deny:.2f}",
            va="center",
            fontsize=6,
            color=MODEL_COLORS[model.model],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[model.model] for model in data.models])
    ax.set_xlim(-0.02, 0.68)
    ax.set_xticks((0, 0.2, 0.4, 0.6))
    ax.set_xlabel("Error rate on stable controls")
    ax.set_title("False denial dominates errors", loc="left", fontweight="bold")
    handles = (
        Line2D([], [], marker="o", ls="none", color="#4D4D4D", label="False deny"),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            mfc="white",
            mec="#4D4D4D",
            color="#4D4D4D",
            label="False admit",
        ),
    )
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.52, 0.73),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0,
    )
    ax.text(
        0,
        -0.42,
        "posthoc descriptive; n = 64 judgments per role/model",
        transform=ax.get_xaxis_transform(),
        fontsize=5.5,
        color="#606060",
    )
    _panel_label(ax, "b")


def _plot_panel_c(ax: plt.Axes, data: FigureData) -> None:
    matrix = np.asarray(
        [[model.axis_overflip[axis] for axis in AXIS_ORDER] for model in data.models]
    )
    cmap = LinearSegmentedColormap.from_list(
        "overflip",
        ("#F7F7FA", "#B4C0E4", "#B64342"),
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= 0.36 else "#272727"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
                fontweight="bold" if value >= 0.4 else "normal",
            )
    ax.set_xticks(range(len(AXIS_ORDER)))
    ax.set_xticklabels([AXIS_LABELS[axis] for axis in AXIS_ORDER], rotation=32, ha="right")
    ax.set_yticks(range(len(data.models)))
    ax.set_yticklabels([MODEL_LABELS[model.model] for model in data.models])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("Overflip rate", fontsize=6.5)
    colorbar.set_ticks((0, 0.25, 0.5))
    colorbar.ax.tick_params(labelsize=6, length=2)
    ax.set_title("Lifecycle intent is weakest", loc="left", fontweight="bold")
    ax.text(
        0,
        -0.42,
        "posthoc descriptive; n = 16 stable pairs per cell",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#606060",
    )
    _panel_label(ax, "c")


def render_figure(data: FigureData, output_dir: Path) -> tuple[Path, ...]:
    """Render fixed-size editable vector and raster publication outputs."""
    apply_style()
    figure = plt.figure(figsize=(7.2, 3.4), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.35, 1.0, 1.15),
        left=0.075,
        right=0.965,
        top=0.88,
        bottom=0.22,
        wspace=0.44,
    )
    _plot_panel_a(figure.add_subplot(grid[0, 0]), data)
    _plot_panel_b(figure.add_subplot(grid[0, 1]), data)
    _plot_panel_c(figure.add_subplot(grid[0, 2]), data)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_dir / f"{FIGURE_BASENAME}.{suffix}" for suffix in ("svg", "pdf", "png"))
    figure.savefig(paths[0], metadata={"Date": None})
    svg = "\n".join(line.rstrip() for line in paths[0].read_text(encoding="utf-8").splitlines())
    _write_text(paths[0], svg + "\n")
    figure.savefig(paths[1], metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(paths[2], dpi=300)
    plt.close(figure)
    return paths


def caption_text() -> str:
    return """# Figure Caption

**Controlled counterfactual pairs separate focal rule-following from selective
verification.** **a,** Strict focal-pair consistency versus stable-control overflip.
Points show three comparison-eligible providers; horizontal and vertical intervals are
95% scenario-bootstrap confidence intervals (10,000 replicates; 16 scenario units,
32 focal pairs, and 64 stable candidate pairs). Dashed boundaries mark the frozen
supportive criteria (focal consistency at least 0.80; stable overflip at most 0.05).
**b,** Posthoc descriptive false-denial rate for stable-admissible memories and
false-admission rate for stable-inadmissible memories (64 judgments per role and model).
**c,** Posthoc stable-control overflip by governing axis (16 stable candidate pairs per
cell). Candidate pools and order are fixed within each pair; only the query condition
changes. Posthoc panels change no prompt, prediction, threshold, gate, or comparison
eligibility. Claude Sonnet 5 is excluded because its full bundle failed the exact output
contract; no prefix was scored.
"""


def _pixel_audit(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    nonwhite = np.any(rgb < 248, axis=2)
    return {
        "width_px": int(rgb.shape[1]),
        "height_px": int(rgb.shape[0]),
        "nonwhite_fraction": float(nonwhite.mean()),
        "channel_min": int(rgb.min()),
        "channel_max": int(rgb.max()),
    }


def generate(results_dir: Path, output_dir: Path) -> Mapping[str, object]:
    """Validate source manifests, render exports, and bind the complete figure package."""
    base_manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    posthoc_manifest = json.loads(
        (results_dir / "posthoc_manifest.json").read_text(encoding="utf-8")
    )
    if base_manifest.get("official_result") is not False:
        raise ValueError("base result boundary drifted")
    if posthoc_manifest.get("provider_call_count") != 0:
        raise ValueError("posthoc result boundary drifted")
    data = load_figure_data(results_dir)
    source_path = output_dir / f"{FIGURE_BASENAME}_source_data.csv"
    _write_csv(source_path, source_data_rows(data))
    caption_path = output_dir / f"{FIGURE_BASENAME}_caption.md"
    _write_text(caption_path, caption_text())
    figure_paths = render_figure(data, output_dir)
    svg_text = figure_paths[0].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise ValueError("SVG text was converted to paths")
    if not figure_paths[1].read_bytes().startswith(b"%PDF"):
        raise ValueError("PDF export is invalid")
    pixel_audit = _pixel_audit(figure_paths[2])
    if pixel_audit["nonwhite_fraction"] < 0.03:
        raise ValueError("PNG export appears blank")
    inputs = (
        "aggregate_metrics.csv",
        "bootstrap_ci.csv",
        "error_taxonomy.csv",
        "stable_control_outcomes.csv",
        "manifest.json",
        "posthoc_manifest.json",
    )
    outputs = (source_path, caption_path, *figure_paths)
    manifest = {
        "schema_version": 1,
        "status": "publication_figure_from_frozen_aggregate_results",
        "core_conclusion": (
            "focal counterfactual success does not establish selective admissibility verification"
        ),
        "archetype": "quantitative_grid",
        "backend": "python_matplotlib",
        "final_width_in": 7.2,
        "final_height_in": 3.4,
        "png_dpi": 300,
        "svg_text_editable": True,
        "panel_a_analysis": "preregistered",
        "panels_b_c_analysis": "posthoc_descriptive",
        "provider_call_count": 0,
        "inputs": [{"path": name, "sha256": _sha256(results_dir / name)} for name in inputs],
        "script_sha256": _sha256(Path(__file__)),
        "outputs": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "png_pixel_audit": pixel_audit,
        "image_integrity": "direct_vector_plot_from_published_csv_no_image_adjustment",
        "official_result": False,
    }
    _write_json(output_dir / f"{FIGURE_BASENAME}_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = generate(args.results_dir, args.output_dir)
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
