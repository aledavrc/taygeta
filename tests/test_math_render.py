"""Tests for statement rendering with TeX formulas (``math_render``).

The formatting preview must show Markdown *and* formulas. ``markdown2`` mangles
TeX (``$x_1$`` → ``$x<em>1</em>$``, ``\\{`` → ``{``, ``\\\\`` → ``\\``,
``&`` → ``&amp;``), so ``render_statement_html`` extracts the formulas before
rendering and reinserts them intact so KaTeX can process them in the browser.
"""

from __future__ import annotations

import html as html_module

from taygeta.ui.nice.math_render import KATEX_CDN, MATH_CLASS, render_statement_html


def _decoded(rendered: str) -> str:
    """Unescape entities as the browser would when parsing the HTML."""
    return html_module.unescape(rendered)


def test_subscripts_are_not_mangled() -> None:
    """``$x_1 + x_2$`` must not become emphasis (original bug)."""
    rendered = render_statement_html("El valor de $x_1 + x_2$ es la suma.")
    assert "<em>" not in rendered
    assert "$x_1 + x_2$" in rendered


def test_math_specials_survive() -> None:
    """TeX special characters survive rendering."""
    statement = r"Sea $\{x \in \mathbb{R} : x > 0\}$ y $a \cdot b = c$."
    decoded = _decoded(render_statement_html(statement))
    assert r"$\{x \in \mathbb{R} : x > 0\}$" in decoded
    assert r"$a \cdot b = c$" in decoded


def test_matrices_and_ampersands_survive() -> None:
    """Matrices (``&`` and ``\\\\``) are neither escaped nor collapsed."""
    statement = r"La matriz $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$."
    decoded = _decoded(render_statement_html(statement))
    assert r"$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$" in decoded


def test_display_math_is_preserved() -> None:
    """The ``$$...$$`` block is preserved so KaTeX renders it in display mode."""
    statement = r"$$ \int_0^1 x^2\,dx = \frac{1}{3} $$"
    decoded = _decoded(render_statement_html(statement))
    assert "$$" in decoded
    assert r"\int_0^1 x^2\,dx = \frac{1}{3}" in decoded


def test_markdown_formatting_still_works() -> None:
    """Text Markdown (bold) still renders."""
    rendered = render_statement_html("Texto **negrita** y $x_1$.")
    assert "<strong>negrita</strong>" in rendered


def test_no_sentinel_leaks() -> None:
    """Internal sentinels must never appear in the output."""
    rendered = render_statement_html("$x_1$ y $y_2$ y texto sin fórmulas")
    assert "\ue000" not in rendered
    assert "\ue001" not in rendered


def test_empty_statement() -> None:
    """An empty statement produces empty HTML."""
    assert render_statement_html("") == ""


def test_katex_assets_and_marker() -> None:
    """The KaTeX CDN and the marker class are the expected ones."""
    assert MATH_CLASS == "taygeta-math"
    assert "katex" in KATEX_CDN
