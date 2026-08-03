from __future__ import annotations

from pathlib import Path

from scripts.export_overleaf import export_overleaf

ROOT = Path(__file__).resolve().parents[1]


def test_overleaf_export_flattens_inputs_and_collects_figures(tmp_path: Path) -> None:
    output = tmp_path / "overleaf"
    written = export_overleaf(ROOT, output)
    main = (output / "main.tex").read_text(encoding="utf-8")

    assert "\\input{" not in main
    assert "generated/" not in main
    assert "../results/" not in main
    assert "\\includegraphics" in main
    assert all("{figure/" in line for line in main.splitlines() if "includegraphics" in line)
    assert {path.name for path in (output / "figure").iterdir()} == {
        "constraint_reliability.pdf",
        "counterfactual_exposure.pdf",
        "evidence_summary.pdf",
        "inference_gap.pdf",
        "retrieval_results.pdf",
        "verification_pipeline.pdf",
    }
    assert {path.relative_to(output).as_posix() for path in written} == {
        "main.tex",
        "references.bib",
        "neurips_2026.sty",
        "figure/constraint_reliability.pdf",
        "figure/counterfactual_exposure.pdf",
        "figure/evidence_summary.pdf",
        "figure/inference_gap.pdf",
        "figure/retrieval_results.pdf",
        "figure/verification_pipeline.pdf",
    }
