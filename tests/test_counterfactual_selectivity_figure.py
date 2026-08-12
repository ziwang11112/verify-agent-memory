from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import plot_counterfactual_selectivity_figure as figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "counterfactual_admissibility"
FIGURES = RESULTS / "figures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_figure_data_matches_frozen_primary_and_posthoc_results() -> None:
    data = figure.load_figure_data(RESULTS)
    models = {model.model: model for model in data.models}

    assert models["OpenAI/gpt-5.6-sol"].focal.value == 1
    assert models["Gemini/gemini-3.6-flash"].focal.value == 1
    assert models["DeepSeek/deepseek-v4-pro"].focal.value == pytest.approx(0.71875)
    assert models["OpenAI/gpt-5.6-sol"].stable_false_deny == pytest.approx(0.265625)
    assert models["Gemini/gemini-3.6-flash"].stable_false_admit == 0
    assert models["DeepSeek/deepseek-v4-pro"].axis_overflip["lifecycle_intent"] == (
        pytest.approx(0.4375)
    )


def test_source_data_labels_preregistered_and_posthoc_panels() -> None:
    rows = figure.source_data_rows(figure.load_figure_data(RESULTS))

    assert len(rows) == 28
    assert {row["analysis_status"] for row in rows if row["panel"] == "a"} == {
        "preregistered",
        "reference",
    }
    assert {row["analysis_status"] for row in rows if row["panel"] in {"b", "c"}} == {
        "posthoc_descriptive"
    }


def test_rendered_figure_has_fixed_canvas_and_editable_svg(tmp_path: Path) -> None:
    manifest = figure.generate(RESULTS, tmp_path)
    png = tmp_path / "counterfactual_selectivity.png"
    svg = tmp_path / "counterfactual_selectivity.svg"
    pdf = tmp_path / "counterfactual_selectivity.pdf"

    with Image.open(png) as image:
        assert image.size == (2160, 1020)
    svg_text = svg.read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert all(line == line.rstrip() for line in svg_text.splitlines())
    assert pdf.read_bytes().startswith(b"%PDF")
    assert manifest["png_pixel_audit"]["nonwhite_fraction"] > 0.03
    assert manifest["provider_call_count"] == 0


def test_rendered_exports_are_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    figure.generate(RESULTS, first)
    figure.generate(RESULTS, second)

    for suffix in ("svg", "pdf", "png"):
        name = f"counterfactual_selectivity.{suffix}"
        assert _sha256(first / name) == _sha256(second / name)


def test_published_figure_manifest_binds_source_and_exports() -> None:
    manifest_path = FIGURES / "counterfactual_selectivity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "publication_figure_from_frozen_aggregate_results"
    assert manifest["backend"] == "python_matplotlib"
    assert manifest["svg_text_editable"] is True
    assert manifest["panel_a_analysis"] == "preregistered"
    assert manifest["panels_b_c_analysis"] == "posthoc_descriptive"
    assert manifest["official_result"] is False
    for output in manifest["outputs"]:
        path = FIGURES / output["path"]
        assert path.is_file()
        assert _sha256(path) == output["sha256"]
        assert path.stat().st_size == output["bytes"]


def test_published_source_data_has_no_case_identifiers_or_raw_text() -> None:
    path = FIGURES / "counterfactual_selectivity_source_data.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "pair_id" not in rows[0]
    assert "scenario_id" not in rows[0]
    assert "query" not in rows[0]
    assert "text" not in rows[0]
