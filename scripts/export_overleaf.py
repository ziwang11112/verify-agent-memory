"""Export a flat Overleaf project with one self-contained main.tex file."""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

INPUT_PATTERN = re.compile(r"(?m)^(?P<indent>[ \t]*)\\input\{(?P<path>[^}]+)\}[ \t]*$")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?P<options>\[[^\]]*\])?\{(?P<path>[^}]+)\}")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes repository: {path}")
    return resolved


def _inline_inputs(
    text: str,
    *,
    base_dir: Path,
    repository_root: Path,
    stack: tuple[Path, ...] = (),
) -> str:
    def replace(match: re.Match[str]) -> str:
        relative = Path(match.group("path"))
        if not relative.suffix:
            relative = relative.with_suffix(".tex")
        source = _inside(base_dir / relative, repository_root)
        if source in stack:
            chain = " -> ".join(path.as_posix() for path in (*stack, source))
            raise ValueError(f"recursive LaTeX input: {chain}")
        if not source.is_file():
            raise FileNotFoundError(source)
        fragment = _inline_inputs(
            source.read_text(encoding="utf-8"),
            base_dir=source.parent,
            repository_root=repository_root,
            stack=(*stack, source),
        ).rstrip()
        label = source.name
        indent = match.group("indent")
        return f"{indent}% BEGIN INLINED {label}\n{fragment}\n{indent}% END INLINED {label}"

    previous = None
    while previous != text:
        previous = text
        text = INPUT_PATTERN.sub(replace, text)
    return text


def export_overleaf(repository_root: Path, output_root: Path) -> tuple[Path, ...]:
    repository_root = repository_root.resolve()
    paper_root = repository_root / "paper"
    main_path = paper_root / "main.tex"
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figure"
    figure_root.mkdir(exist_ok=True)

    main = _inline_inputs(
        main_path.read_text(encoding="utf-8"),
        base_dir=paper_root,
        repository_root=repository_root,
        stack=(main_path.resolve(),),
    )
    copied_figures: dict[str, Path] = {}

    def replace_graphic(match: re.Match[str]) -> str:
        relative = Path(match.group("path"))
        source = _inside(paper_root / relative, repository_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        existing = copied_figures.get(source.name)
        if existing is not None and existing != source:
            raise ValueError(f"duplicate figure basename: {source.name}")
        copied_figures[source.name] = source
        options = match.group("options") or ""
        return f"\\includegraphics{options}{{figure/{source.name}}}"

    main = GRAPHICS_PATTERN.sub(replace_graphic, main)
    if "\\input{" in main:
        raise ValueError("exported main.tex still contains an input directive")

    output_main = output_root / "main.tex"
    output_main.write_text(main, encoding="utf-8", newline="\n")
    written = [output_main]
    for name in ("references.bib", "neurips_2026.sty"):
        destination = output_root / name
        shutil.copy2(paper_root / name, destination)
        written.append(destination)
    for name, source in sorted(copied_figures.items()):
        destination = figure_root / name
        shutil.copy2(source, destination)
        written.append(destination)
    return tuple(written)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "tmp" / "overleaf_export",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    written = export_overleaf(args.repository_root, args.output)
    print(f"exported {len(written)} files to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
