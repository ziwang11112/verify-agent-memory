from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import plot_counterfactual_exposure_figure as figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "counterfactual_exposure"
FIGURES = RESULTS / "figures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_figure_data_matches_frozen_exposure_results() -> None:
    models = {model.provider: model for model in figure.load_figure_data(RESULTS)}

    assert models["OpenAI"].cells["relevant_admissible"].effect.value == pytest.approx(0.90625)
    assert models["Gemini"].selectivity_gap.value == pytest.approx(0.8125)
    deepseek_risk = models["DeepSeek"].cells["relevant_inadmissible"].effect
    assert deepseek_risk.value == pytest.approx(0.15625)
    assert deepseek_risk.lower == pytest.approx(0.03125)
    assert deepseek_risk.upper == pytest.approx(0.3125)


def test_source_data_contains_every_plotted_quantity() -> None:
    rows = figure.source_data_rows(figure.load_figure_data(RESULTS))

    assert len(rows) == 18
    assert {row["panel"] for row in rows} == {"a", "b"}
    assert {row["provider"] for row in rows} == {"OpenAI", "Gemini", "DeepSeek"}


def test_rendered_figure_has_fixed_canvas_and_editable_svg(tmp_path: Path) -> None:
    manifest = figure.generate(RESULTS, tmp_path)
    png = tmp_path / "counterfactual_exposure.png"
    svg = tmp_path / "counterfactual_exposure.svg"
    pdf = tmp_path / "counterfactual_exposure.pdf"

    with Image.open(png) as image:
        assert image.size == (2160, 960)
    assert "<text" in svg.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert manifest["png_pixel_audit"]["nonwhite_fraction"] > 0.03
    assert manifest["provider_call_count"] == 0
    assert manifest["model_pooling"] is False


def test_rendered_exports_are_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    figure.generate(RESULTS, first)
    figure.generate(RESULTS, second)

    for suffix in ("svg", "pdf", "png"):
        name = f"counterfactual_exposure.{suffix}"
        assert _sha256(first / name) == _sha256(second / name)


def test_published_figure_manifest_binds_source_and_exports() -> None:
    manifest = json.loads(
        (FIGURES / "counterfactual_exposure_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "publication_figure_from_controlled_content_free_results"
    assert manifest["svg_text_editable"] is True
    assert manifest["official_result"] is False
    for output in manifest["outputs"]:
        path = FIGURES / output["path"]
        assert path.is_file()
        assert _sha256(path) == output["sha256"]
        assert path.stat().st_size == output["bytes"]


def test_published_source_data_contains_no_text_or_unit_identifiers() -> None:
    with (FIGURES / "counterfactual_exposure_source_data.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    forbidden = {"answer", "prompt", "query_text", "response", "pair_id", "scenario_id"}
    assert not set(rows[0]).intersection(forbidden)
