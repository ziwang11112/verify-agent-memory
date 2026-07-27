"""Compile the paper with pdflatex and BibTeX without requiring latexmk."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


def _run(command: Sequence[str], *, working_directory: Path) -> None:
    subprocess.run(command, cwd=working_directory, check=True)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    paper_root = repository_root / "paper"
    build_root = paper_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None or bibtex is None:
        raise SystemExit("pdflatex and bibtex must both be available on PATH")

    latex_command = (
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-output-directory=build",
        "main.tex",
    )
    _run(latex_command, working_directory=paper_root)
    _run((bibtex, "build/main"), working_directory=paper_root)
    _run(latex_command, working_directory=paper_root)
    _run(latex_command, working_directory=paper_root)
    print(f"compiled {build_root / 'main.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
