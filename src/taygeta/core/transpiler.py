"""Markdown/TeX to Typst and LaTeX transpiler (section 10).

Design (no global regex substitutions):

* ``pandoc_to_ast`` invokes the Pandoc CLI via ``subprocess`` (list arguments,
  never ``shell=True``) and returns the document's JSON AST.
* ``_render_blocks`` / ``_render_inlines`` recursively walk the block and inline
  nodes preserving the semantic structure.
* ``Math`` nodes (``InlineMath``/``DisplayMath``) are processed with
  :class:`MathConverter`, a facade over **tylax** (a LaTeX↔Typst converter
  written in Rust): every TeX formula is delegated to ``tylax.latex_to_typst``
  with ``frac_to_slash=False`` and ``non_strict=True``.
* ``{{fig:N}}`` placeholders are detected with a manual scanner (no regex) and
  become ``#figure(image("assets/...", width: 80%))`` blocks.
* LaTeX export uses Pandoc (``--to latex``) and ``\\includegraphics`` for the
  figures.
* ``compile_typst`` / ``validate_typst`` run the real ``typst`` binary in
  compilation mode and return the errors with context.

Note on error handling: tylax does **not** raise on unknown TeX commands; with
``non_strict=True`` it keeps them as raw text (in strict mode it silently drops
them), so an unknown command is detected in the real Typst compilation
(``unknown variable: ...``) and a broken exam never produces a PDF without
warning. LaTeX that tylax cannot parse produces the marker
``/* LaTeX Error ... */``, which :meth:`MathConverter.convert` raises as
:class:`TranspileError`, preserving the original TeX for recovery or LaTeX
export.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tylax

from taygeta.core.schema import sanitize_math_delimiters, sanitize_norm_bars

__all__ = [
    "CompileResult",
    "MathConverter",
    "TranspileError",
    "compile_typst",
    "markdown_to_latex",
    "markdown_to_typst",
    "pandoc_to_ast",
    "sanitize_math_for_typst",
    "validate_typst",
]


class TranspileError(RuntimeError):
    """Controlled conversion error, preserving the original TeX when applicable."""

    def __init__(self, message: str, *, original: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original = original


@dataclass(slots=True)
class CompileResult:
    """Result of a real Typst compilation."""

    ok: bool
    pdf_path: Path | None
    stdout: str
    stderr: str
    returncode: int

    def format_errors(self) -> str:
        """Return the Typst error message with context, if there was one."""
        return self.stderr.strip() or "no errors"


# ---------------------------------------------------------------------------
# Pandoc
# ---------------------------------------------------------------------------


def pandoc_to_ast(markdown: str, *, pandoc_bin: str = "pandoc") -> dict[str, Any]:
    """Get the Pandoc JSON AST for a Markdown document.

    Raises:
        TranspileError: if Pandoc is not available, fails or returns invalid JSON.
    """
    executable = shutil.which(pandoc_bin) or pandoc_bin
    process = subprocess.run(
        [executable, "--from=markdown", "--to=json"],
        input=markdown,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise TranspileError(f"pandoc failed: {process.stderr.strip()}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise TranspileError(f"pandoc returned invalid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# TeX -> Typst converter (delegated to tylax)
# ---------------------------------------------------------------------------


#: Fixed conversion options for the bank formulas.
#:
#: * ``frac_to_slash=False`` keeps ``frac(a, b)`` (style of the previous
#:   converter) instead of the ``a/b`` shortcut.
#: * ``non_strict=True`` (tylax's default) keeps unknown commands as raw text
#:   instead of silently dropping them: this way the final ``typst compile``
#:   detects them and a broken statement never produces a PDF without warning
#:   (a domain invariant).
_L2T_OPTIONS = tylax.L2TOptions(frac_to_slash=False, non_strict=True)

#: Marker that tylax inserts when the LaTeX input could not be parsed.
_LATEX_ERROR_MARK = "/* LaTeX Error"


class MathConverter:
    """Structural converter of a TeX formula to Typst (a facade over tylax).

    tylax (a LaTeX↔Typst converter written in Rust) translates each formula with
    a symbol catalog and macro expansion far broader than the homegrown parser it
    replaces. The class keeps the previous public API (``convert(tex)``) and the
    ``_original`` attribute so the module's block/inline renderers do not change.
    """

    def __init__(self) -> None:
        self._original = ""

    def convert(self, tex: str) -> str:
        """Convert a TeX formula (without ``$`` delimiters) to Typst.

        Raises:
            TranspileError: if tylax could not parse the input LaTeX (the output
                contains the ``/* LaTeX Error ... */`` marker). Unknown commands
                do **not** raise: they are kept as raw text and the real Typst
                compilation will detect them.
        """
        self._original = tex
        result = tylax.latex_to_typst(tex, options=_L2T_OPTIONS)
        if _LATEX_ERROR_MARK in result:
            raise TranspileError(f"LaTeX not parseable by tylax: {result.strip()}", original=tex)
        return result


# ---------------------------------------------------------------------------
# Figure placeholders (manual scanner, no regex)
# ---------------------------------------------------------------------------


def _scan_figure_placeholders(text: str) -> list[tuple[str, str]]:
    """Return a list of ``(kind, value)`` pairs: ``("text", ...)`` or ``("figure", n)``."""
    spans: list[tuple[str, str]] = []
    marker = "{{fig:"
    index = 0
    while index < len(text):
        found = text.find(marker, index)
        if found == -1:
            spans.append(("text", text[index:]))
            break
        if found > index:
            spans.append(("text", text[index:found]))
        cursor = found + len(marker)
        digits = ""
        while cursor < len(text) and text[cursor].isdigit():
            digits += text[cursor]
            cursor += 1
        if digits and cursor + 1 < len(text) and text[cursor : cursor + 2] == "}}":
            spans.append(("figure", digits))
            index = cursor + 2
        else:
            spans.append(("text", text[index : index + 1]))
            index = index + 1
    return spans


def _figure_asset(number: str, figure_paths: Mapping[str, str]) -> str:
    key = f"fig:{number}"
    return figure_paths.get(key, f"fig_{number}.png")


def _figure_block(number: str, figure_paths: Mapping[str, str]) -> str:
    asset = _figure_asset(number, figure_paths)
    return f'#figure(image("assets/{asset}", width: 80%))'


# ---------------------------------------------------------------------------
# Rendering of the Pandoc AST
# ---------------------------------------------------------------------------


def _render_str(text: str, figure_paths: Mapping[str, str]) -> str:
    parts: list[str] = []
    for kind, value in _scan_figure_placeholders(text):
        if kind == "figure":
            parts.append("\n\n" + _figure_block(value, figure_paths) + "\n\n")
        else:
            parts.append(value)
    return "".join(parts)


def _render_inlines(
    inlines: list[dict[str, Any]], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    parts: list[str] = []
    for node in inlines:
        node_type = node.get("t")
        content = node.get("c")
        if node_type == "Str":
            parts.append(_render_str(content, figure_paths))
        elif node_type == "Space":
            parts.append(" ")
        elif node_type == "SoftBreak":
            parts.append(" ")
        elif node_type == "LineBreak":
            parts.append("\n")
        elif node_type == "Emph":
            parts.append("_" + _render_inlines(content, converter, figure_paths) + "_")
        elif node_type == "Strong":
            parts.append("*" + _render_inlines(content, converter, figure_paths) + "*")
        elif node_type == "Code":
            parts.append("`" + content[1] + "`")
        elif node_type == "Link":
            text = _render_inlines(content[1], converter, figure_paths)
            parts.append(f'#link("{content[2][0]}")[{text}]')
        elif node_type == "Math":
            parts.append(_render_math(content, converter))
        elif node_type == "RawInline":
            parts.append(_render_raw_inline(content, converter))
        elif node_type == "Quoted":
            parts.append('"' + _render_inlines(content[1], converter, figure_paths) + '"')
        elif node_type == "Span":
            parts.append(_render_inlines(content[1], converter, figure_paths))
        elif node_type == "Note":
            continue
        else:
            raise TranspileError(f"unsupported inline element: {node_type}")
    return "".join(parts)


def _render_math(content: list[Any], converter: MathConverter) -> str:
    math_type = content[0].get("t")
    tex = content[1]
    typst = converter.convert(tex)
    if math_type == "DisplayMath":
        return "\n$ " + typst + " $\n"
    return "$" + typst + "$"


def _render_raw_inline(content: list[Any], converter: MathConverter) -> str:
    fmt = content[0]
    text = content[1]
    if fmt in ("tex", "latex"):
        return "$" + converter.convert(text) + "$"
    return text


def _render_list_item(
    item: list[dict[str, Any]], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    return _render_blocks(item, converter, figure_paths).strip()


def _render_block(
    block: dict[str, Any], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    node_type = block.get("t")
    content = block.get("c")
    if node_type in ("Para", "Plain"):
        return _render_inlines(content, converter, figure_paths)
    if node_type == "Header":
        level = content[0]
        text = _render_inlines(content[2], converter, figure_paths)
        return "=" * level + " " + text
    if node_type == "BulletList":
        items = ["- " + _render_list_item(item, converter, figure_paths) for item in content]
        return "\n".join(items)
    if node_type == "OrderedList":
        rendered = ["+ " + _render_list_item(item, converter, figure_paths) for item in content[1]]
        return "\n".join(rendered)
    if node_type == "CodeBlock":
        return "```\n" + content[1] + "\n```"
    if node_type == "BlockQuote":
        inner = _render_blocks(content, converter, figure_paths)
        return "\n".join("> " + line for line in inner.splitlines())
    if node_type == "HorizontalRule":
        return ""
    if node_type == "Div":
        return _render_blocks(content[1], converter, figure_paths)
    if node_type == "Null":
        return ""
    raise TranspileError(f"unsupported block: {node_type}")


def _render_blocks(
    blocks: list[dict[str, Any]], converter: MathConverter, figure_paths: Mapping[str, str]
) -> str:
    rendered = [_render_block(block, converter, figure_paths) for block in blocks]
    return "\n\n".join(part for part in rendered if part)


# ---------------------------------------------------------------------------
# Math sanitization for Typst
# ---------------------------------------------------------------------------


def _replace_empty_identifier(tex: str) -> str:
    """Rewrite the word ``empty`` as ``\\emptyset`` inside a formula.

    Some OCR/LLM models write ``empty`` (the name of the empty set in other
    syntaxes) where Typst expects ``emptyset``; the transpiler does not recognize
    that identifier and Typst aborts with ``unknown variable: empty``. Only the
    whole word is replaced, so ``\\emptyset`` (whose identifier is ``emptyset``,
    not ``empty``) stays intact. The invalid command ``\\empty`` is also
    normalized to ``\\emptyset``.
    """
    parts: list[str] = []
    index = 0
    length = len(tex)
    while index < length:
        char = tex[index]
        if char.isalpha():
            end = index
            while end < length and tex[end].isalnum():
                end += 1
            word = tex[index:end]
            parts.append(r"\emptyset" if word == "empty" else word)
            index = end
            continue
        if char == "\\" and index + 1 < length and tex[index + 1].isalpha():
            end = index + 1
            while end < length and tex[end].isalpha():
                end += 1
            command = tex[index + 1 : end]
            parts.append(r"\emptyset" if command == "empty" else tex[index:end])
            index = end
            continue
        parts.append(char)
        index += 1
    return "".join(parts)


def _find_inline_math_close(text: str, start: int) -> int:
    """Return the index of the ``$`` that closes an inline formula (or ``-1``).

    Escaped dollars (``\\$``) that may appear inside the formula are ignored.
    """
    index = start
    length = len(text)
    while index < length:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "$":
            return index
        index += 1
    return -1


def _rewrite_empty_set(markdown: str) -> str:
    """Replace ``empty`` with ``\\emptyset`` inside the math blocks.

    It walks the Markdown distinguishing display (``$$...$$``) from inline
    (``$...$``) and applies :func:`_replace_empty_identifier` only inside the
    formulas, so the English word ``empty`` in normal text is never altered.
    """
    parts: list[str] = []
    index = 0
    length = len(markdown)
    while index < length:
        char = markdown[index]
        if char == "\\" and index + 1 < length and markdown[index + 1] == "$":
            parts.append(markdown[index : index + 2])
            index += 2
            continue
        if char != "$":
            parts.append(char)
            index += 1
            continue
        if markdown.startswith("$$", index):
            close = markdown.find("$$", index + 2)
            if close == -1:
                parts.append(markdown[index:])
                break
            parts.append("$$")
            parts.append(_replace_empty_identifier(markdown[index + 2 : close]))
            parts.append("$$")
            index = close + 2
            continue
        close = _find_inline_math_close(markdown, index + 1)
        if close == -1:
            parts.append(markdown[index:])
            break
        parts.append("$")
        parts.append(_replace_empty_identifier(markdown[index + 1 : close]))
        parts.append("$")
        index = close + 1
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_math_for_typst(markdown: str) -> str:
    """Sanitize the input Markdown right before compiling to Typst.

    Defensive compiler hardening: although the schemas already normalize the
    delimiters on validation (and the database stores the canonical form), this
    step guarantees that any statement reaching the transpiler is ready for Typst:

    * ``\\[...\\]`` and ``\\(...\\)`` are rewritten to ``$$...$$`` / ``$...$``
      (with :func:`~taygeta.core.schema.sanitize_math_delimiters`), so Pandoc
      generates ``DisplayMath``/``InlineMath`` nodes and the rendering produces
      Typst's block delimiters ``$ ... $``;
    * ``\\|`` (LaTeX norm bar) is rewritten as ``||`` (Typst norm) with
      :func:`~taygeta.core.schema.sanitize_norm_bars`;
    * the word ``empty`` inside formulas is rewritten as ``\\emptyset`` (the
      empty-set symbol) to avoid the Typst error ``unknown variable: empty``.
    """
    sanitized = sanitize_norm_bars(sanitize_math_delimiters(markdown))
    return _rewrite_empty_set(sanitized)


def markdown_to_typst(
    markdown: str,
    *,
    figure_paths: Mapping[str, str] | None = None,
    pandoc_bin: str = "pandoc",
) -> str:
    """Convert Markdown (with TeX ``$...$``/``$$...$$``) to Typst.

    Args:
        markdown: Markdown document with TeX math and ``{{fig:N}}`` placeholders.
        figure_paths: Optional mapping ``fig:id`` → asset name relative to ``assets/``.
        pandoc_bin: Path of the Pandoc executable.

    Raises:
        TranspileError: if there is an unsupported block/inline/construct.
    """
    sanitized = sanitize_math_for_typst(markdown)
    ast = pandoc_to_ast(sanitized, pandoc_bin=pandoc_bin)
    converter = MathConverter()
    return _render_blocks(ast.get("blocks", []), converter, figure_paths or {}) + "\n"


_LATEX_FIG_SENTINEL = "TAYGETAFIG"


def _protect_figures(markdown: str) -> tuple[str, dict[str, str]]:
    """Replace ``{{fig:N}}`` with alphanumeric sentinels before Pandoc.

    Pandoc would escape the braces in the LaTeX output (``\\{\\{fig:1\\}\\}``), so
    the placeholders are replaced with alphanumeric tokens that Pandoc respects.
    """
    parts: list[str] = []
    mapping: dict[str, str] = {}
    for kind, value in _scan_figure_placeholders(markdown):
        if kind == "figure":
            sentinel = _LATEX_FIG_SENTINEL + value
            mapping[sentinel] = value
            parts.append(sentinel)
        else:
            parts.append(value)
    return "".join(parts), mapping


def _replace_latex_sentinels(
    text: str, mapping: dict[str, str], figure_paths: Mapping[str, str]
) -> str:
    parts: list[str] = []
    index = 0
    while index < len(text):
        found = text.find(_LATEX_FIG_SENTINEL, index)
        if found == -1:
            parts.append(text[index:])
            break
        parts.append(text[index:found])
        cursor = found + len(_LATEX_FIG_SENTINEL)
        digits = ""
        while cursor < len(text) and text[cursor].isdigit():
            digits += text[cursor]
            cursor += 1
        sentinel = _LATEX_FIG_SENTINEL + digits
        if sentinel in mapping:
            asset = _figure_asset(mapping[sentinel], figure_paths)
            parts.append(
                "\\begin{center}\n\\includegraphics[width=0.8\\linewidth]"
                f"{{assets/{asset}}}\n\\end{{center}}"
            )
            index = cursor
        else:
            parts.append(text[index : index + 1])
            index += 1
    return "".join(parts)


def markdown_to_latex(
    markdown: str,
    *,
    figure_paths: Mapping[str, str] | None = None,
    pandoc_bin: str = "pandoc",
) -> str:
    """Convert Markdown to LaTeX using Pandoc and ``\\includegraphics`` for figures."""
    protected, mapping = _protect_figures(markdown)
    executable = shutil.which(pandoc_bin) or pandoc_bin
    process = subprocess.run(
        [executable, "--from=markdown", "--to=latex"],
        input=protected,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise TranspileError(f"pandoc failed to export LaTeX: {process.stderr.strip()}")
    return _replace_latex_sentinels(process.stdout, mapping, figure_paths or {})


def compile_typst(
    source: str,
    workdir: str | Path,
    *,
    typst_bin: str = "typst",
    filename: str = "main.typ",
    output_name: str = "output.pdf",
) -> CompileResult:
    """Compile a Typst source to PDF in ``workdir`` using the real binary.

    It does not use ``shell=True``: it invokes ``typst compile`` with list
    arguments.
    """
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    main_file = directory / filename
    main_file.write_text(source)
    executable = shutil.which(typst_bin) or typst_bin
    output = directory / output_name
    process = subprocess.run(
        [executable, "compile", str(main_file), str(output)],
        text=True,
        capture_output=True,
        cwd=str(directory),
    )
    return CompileResult(
        ok=process.returncode == 0,
        pdf_path=output if process.returncode == 0 else None,
        stdout=process.stdout,
        stderr=process.stderr,
        returncode=process.returncode,
    )


def validate_typst(
    source: str,
    *,
    typst_bin: str = "typst",
    filename: str = "main.typ",
) -> CompileResult:
    """Validate a Typst source by compiling it in a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="taygeta-typst-") as directory:
        return compile_typst(
            source, directory, typst_bin=typst_bin, filename=filename, output_name="output.pdf"
        )
