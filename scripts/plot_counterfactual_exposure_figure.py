"""Generate the publication figure for the paired exposure intervention."""

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
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "counterfactual_exposure"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures"
FIGURE_BASENAME = "counterfactual_exposure"

MODEL_ORDER = (
    ("OpenAI", "gpt-5.6-sol"),
    ("Gemini", "gemini-3.6-flash"),
    ("DeepSeek", "deepseek-v4-pro"),
)
MODEL_LABELS = {
    "OpenAI": "GPT-5.6 Sol",
    "Gemini": "Gemini 3.6 Flash",
    "DeepSeek": "DeepSeek V4 Pro",
}
MODEL_COLORS = {
    "OpenAI": "#195B9A",
    "Gemini": "#9A4D8E",
    "DeepSeek": "#248A8D",
}
CELL_ORDER = (
    "relevant_admissible",
    "relevant_inadmissible",
    "irrelevant_admissible",
    "irrelevant_inadmissible",
)
CELL_LABELS = {
    "relevant_admissible": "Relevant + admissible",
    "relevant_inadmissible": "Relevant + inadmissible",
    "irrelevant_admissible": "Irrelevant + admissible",
    "irrelevant_inadmissible": "Irrelevant + inadmissible",
}
GAP_CONTRAST = "relevant_admissible_minus_relevant_inadmissible_exposure_effect"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("source data must be nonempty")
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
        if not -1 <= self.lower <= self.value <= self.upper <= 1:
            raise ValueError("effect interval must be ordered within [-1, 1]")


@dataclass(frozen=True)
class CellData:
    exposed_disclosure: float
    withheld_disclosure: float
    effect: Interval


@dataclass(frozen=True)
class ModelData:
    provider: str
    model: str
    cells: Mapping[str, CellData]
    selectivity_gap: Interval


def load_figure_data(results_dir: Path) -> tuple[ModelData, ...]:
    """Load only the published, content-free aggregate result tables."""
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("official_result") is not False:
        raise ValueError("result boundary drifted")
    if manifest.get("model_pooling") is not False:
        raise ValueError("provider results must not be pooled")
    cells = _read_csv(results_dir / "cell_metrics.csv")
    intervals = _read_csv(results_dir / "bootstrap_ci.csv")
    models = []
    for provider, model in MODEL_ORDER:
        cell_data = {}
        for cell in CELL_ORDER:
            row = _one(cells, provider=provider, model=model, scope="overall", cell=cell)
            interval = _one(
                intervals,
                provider=provider,
                model=model,
                scope="overall",
                contrast=cell,
            )
            cell_data[cell] = CellData(
                exposed_disclosure=float(row["exposed_disclosure_rate"]),
                withheld_disclosure=float(row["withheld_disclosure_rate"]),
                effect=Interval(
                    value=float(interval["estimate"]),
                    lower=float(interval["ci_lower"]),
                    upper=float(interval["ci_upper"]),
                ),
            )
        gap = _one(
            intervals,
            provider=provider,
            model=model,
            scope="overall",
            contrast=GAP_CONTRAST,
        )
        models.append(
            ModelData(
                provider=provider,
                model=model,
                cells=cell_data,
                selectivity_gap=Interval(
                    value=float(gap["estimate"]),
                    lower=float(gap["ci_lower"]),
                    upper=float(gap["ci_upper"]),
                ),
            )
        )
    return tuple(models)


def source_data_rows(models: Sequence[ModelData]) -> tuple[dict[str, object], ...]:
    """Return tidy source data for every plotted quantity."""
    rows = []
    for model in models:
        for cell in CELL_ORDER:
            data = model.cells[cell]
            for exposure, value in (
                ("withheld", data.withheld_disclosure),
                ("exposed", data.exposed_disclosure),
            ):
                rows.append(
                    {
                        "panel": "a",
                        "provider": model.provider,
                        "model": model.model,
                        "cell": cell,
                        "metric": "disclosure_rate",
                        "exposure": exposure,
                        "value": value,
                        "ci_lower": "",
                        "ci_upper": "",
                        "unit_count": 32 if cell.startswith("relevant_") else 64,
                    }
                )
            rows.append(
                {
                    "panel": "b",
                    "provider": model.provider,
                    "model": model.model,
                    "cell": cell,
                    "metric": "paired_exposure_effect",
                    "exposure": "exposed_minus_withheld",
                    "value": data.effect.value,
                    "ci_lower": data.effect.lower,
                    "ci_upper": data.effect.upper,
                    "unit_count": 32 if cell.startswith("relevant_") else 64,
                }
            )
        rows.append(
            {
                "panel": "c",
                "provider": model.provider,
                "model": model.model,
                "cell": "relevant_admissible_minus_relevant_inadmissible",
                "metric": "selectivity_gap",
                "exposure": "difference_in_paired_effects",
                "value": model.selectivity_gap.value,
                "ci_lower": model.selectivity_gap.lower,
                "ci_upper": model.selectivity_gap.upper,
                "unit_count": 64,
            }
        )
    return tuple(rows)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "svg.hashsalt": "counterfactual-exposure-v1",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 6.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _panel_a(axis: plt.Axes, models: Sequence[ModelData]) -> None:
    admissible_color = "#238B57"
    inadmissible_color = "#C44E52"
    for index, model in enumerate(models):
        for offset, cell, color, marker in (
            (-0.14, "relevant_admissible", admissible_color, "o"),
            (0.14, "relevant_inadmissible", inadmissible_color, "s"),
        ):
            data = model.cells[cell]
            x = index + offset
            axis.plot(
                (x, x),
                (data.withheld_disclosure, data.exposed_disclosure),
                color=color,
                linewidth=1.6,
                alpha=0.8,
                zorder=1,
            )
            axis.scatter(
                x,
                data.withheld_disclosure,
                marker=marker,
                s=24,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
            axis.scatter(
                x,
                data.exposed_disclosure,
                marker=marker,
                s=29,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            axis.text(
                x,
                min(1.02, data.exposed_disclosure + 0.055),
                f"{data.exposed_disclosure:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.7,
                color=color,
                fontweight="bold",
            )
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels(
        [MODEL_LABELS[model.provider] for model in models], rotation=18, ha="right"
    )
    axis.set_ylim(-0.03, 1.08)
    axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.set_ylabel("Target disclosure rate")
    axis.set_title("Exposure drives admissible disclosure", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E6E8EA", linewidth=0.6)
    handles = (
        Line2D([], [], marker="o", color=admissible_color, label="Relevant + admissible"),
        Line2D([], [], marker="s", color=inadmissible_color, label="Relevant + inadmissible"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#666666",
            label="Withheld",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            label="Exposed",
        ),
    )
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    _panel_label(axis, "a")


def _panel_b(axis: plt.Axes, models: Sequence[ModelData]) -> None:
    y_base = np.arange(len(CELL_ORDER))[::-1]
    offsets = (0.18, 0.0, -0.18)
    axis.axhspan(2.55, 3.45, color="#EAF5EE", zorder=-3)
    axis.axhspan(1.55, 2.45, color="#FCEEEF", zorder=-3)
    axis.axvline(0, color="#6D7379", linewidth=0.8, linestyle="--", zorder=0)
    for offset, model in zip(offsets, models, strict=True):
        color = MODEL_COLORS[model.provider]
        for base, cell in zip(y_base, CELL_ORDER, strict=True):
            interval = model.cells[cell].effect
            axis.errorbar(
                interval.value,
                base + offset,
                xerr=[
                    [interval.value - interval.lower],
                    [interval.upper - interval.value],
                ],
                fmt="o",
                markersize=4.3,
                capsize=2.0,
                linewidth=1.0,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
    axis.set_yticks(y_base)
    axis.set_yticklabels([CELL_LABELS[cell] for cell in CELL_ORDER])
    axis.set_xlim(-0.22, 1.08)
    axis.set_xticks((-0.2, 0, 0.4, 0.8, 1.0))
    axis.set_xlabel("Paired exposure effect (exposed minus withheld)")
    axis.set_title("Paired effects by evidence cell", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E6E8EA", linewidth=0.6)
    handles = tuple(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=MODEL_COLORS[model.provider],
            label=MODEL_LABELS[model.provider],
        )
        for model in models
    )
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=3,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    _panel_label(axis, "b")


def _panel_c(axis: plt.Axes, models: Sequence[ModelData]) -> None:
    y = np.arange(len(models))[::-1]
    axis.axvline(0, color="#6D7379", linewidth=0.8, linestyle="--", zorder=0)
    for position, model in zip(y, models, strict=True):
        gap = model.selectivity_gap
        risk = model.cells["relevant_inadmissible"].effect
        color = MODEL_COLORS[model.provider]
        axis.errorbar(
            gap.value,
            position + 0.13,
            xerr=[[gap.value - gap.lower], [gap.upper - gap.value]],
            fmt="o",
            markersize=5.4,
            capsize=2.3,
            linewidth=1.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        axis.text(
            min(1.01, gap.upper + 0.035),
            position + 0.13,
            f"{gap.value:.2f}",
            ha="left",
            va="center",
            fontsize=6.0,
            color=color,
            fontweight="bold",
        )
        axis.errorbar(
            risk.value,
            position - 0.13,
            xerr=[[risk.value - risk.lower], [risk.upper - risk.value]],
            fmt="s",
            markersize=4.2,
            capsize=2.0,
            linewidth=1.0,
            color="#C44E52",
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
    axis.set_yticks(y)
    axis.set_yticklabels([MODEL_LABELS[model.provider] for model in models])
    axis.set_xlim(-0.18, 1.10)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.set_xlabel("Paired effect")
    axis.set_title("Selectivity vs residual risk", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E6E8EA", linewidth=0.6)
    handles = (
        Line2D([], [], marker="o", linestyle="none", color="#555555", label="Selectivity gap"),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#C44E52",
            color="#C44E52",
            label="Inadmissible ATE",
        ),
    )
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=1,
        handletextpad=0.35,
    )
    _panel_label(axis, "c")


def render_figure(models: Sequence[ModelData], output_dir: Path) -> tuple[Path, ...]:
    apply_style()
    figure = plt.figure(figsize=(7.2, 3.35), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.18, 1.55, 1.0),
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.27,
        wspace=0.62,
    )
    _panel_a(figure.add_subplot(grid[0, 0]), models)
    _panel_b(figure.add_subplot(grid[0, 1]), models)
    _panel_c(figure.add_subplot(grid[0, 2]), models)
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

**Controlled exposure changes disclosure selectively, but readers remain an unsafe
last line of defense.** **a,** Target-disclosure rates when the manipulated relevant
candidate is withheld (open marker) or exposed (filled marker), separately for
construction-defined admissible and inadmissible candidates. **b,** Paired
exposed-minus-withheld disclosure effects in the four relevance-by-admissibility cells.
Points are reader-specific estimates; whiskers are 95% scenario-bootstrap intervals
(10,000 replicates; 16 scenarios). **c,** Difference between the relevant-admissible
and relevant-inadmissible exposure effects. Readers are reported separately and never
pooled. Red squares show the relevant-inadmissible exposure effect; its interval is
strictly positive for DeepSeek V4 Pro. Each reader completed 384 stateless requests (192 paired
units), with no judge, retry, output repair, or selective rerun. The construction is a
controlled prompt-level diagnostic, not an official benchmark or natural-corpus
prevalence estimate.
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
    """Render exports and bind them to the published content-free result bundle."""
    models = load_figure_data(results_dir)
    source_path = output_dir / f"{FIGURE_BASENAME}_source_data.csv"
    _write_csv(source_path, source_data_rows(models))
    caption_path = output_dir / f"{FIGURE_BASENAME}_caption.md"
    _write_text(caption_path, caption_text())
    figure_paths = render_figure(models, output_dir)
    if "<text" not in figure_paths[0].read_text(encoding="utf-8"):
        raise ValueError("SVG text was converted to paths")
    if not figure_paths[1].read_bytes().startswith(b"%PDF"):
        raise ValueError("PDF export is invalid")
    pixel_audit = _pixel_audit(figure_paths[2])
    if pixel_audit["nonwhite_fraction"] < 0.03:
        raise ValueError("PNG export appears blank")
    inputs = ("cell_metrics.csv", "bootstrap_ci.csv", "manifest.json")
    outputs = (source_path, caption_path, *figure_paths)
    manifest = {
        "schema_version": 1,
        "status": "publication_figure_from_controlled_content_free_results",
        "core_conclusion": (
            "exposure increases disclosure more for relevant admissible than relevant "
            "inadmissible evidence across three separately reported readers"
        ),
        "archetype": "quantitative_grid",
        "backend": "python_matplotlib",
        "final_width_in": 7.2,
        "final_height_in": 3.35,
        "png_dpi": 300,
        "svg_text_editable": True,
        "bootstrap_replicates": 10000,
        "provider_call_count": 0,
        "model_pooling": False,
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
    manifest = generate(args.results_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
