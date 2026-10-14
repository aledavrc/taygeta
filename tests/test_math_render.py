"""Pruebas del renderizado del enunciado con fórmulas TeX (``math_render``).

El recuadro de formato debe mostrar Markdown *y* fórmulas. ``markdown2`` destroza
el TeX (``$x_1$`` → ``$x<em>1</em>$``, ``\\{`` → ``{``, ``\\\\`` → ``\\``,
``&`` → ``&amp;``), así que ``render_statement_html`` extrae las fórmulas antes de
renderizar y las reinserta intactas para que KaTeX las procese en el navegador.
"""

from __future__ import annotations

import html as html_module

from taygeta.ui.nice.math_render import KATEX_CDN, MATH_CLASS, render_statement_html


def _decoded(rendered: str) -> str:
    """Des-escapa las entidades como haría el navegador al parsear el HTML."""
    return html_module.unescape(rendered)


def test_subscripts_are_not_mangled() -> None:
    """``$x_1 + x_2$`` no debe convertirse en énfasis (fallo original)."""
    rendered = render_statement_html("El valor de $x_1 + x_2$ es la suma.")
    assert "<em>" not in rendered
    assert "$x_1 + x_2$" in rendered


def test_math_specials_survive() -> None:
    """Los caracteres especiales de TeX sobreviven al renderizado."""
    statement = r"Sea $\{x \in \mathbb{R} : x > 0\}$ y $a \cdot b = c$."
    decoded = _decoded(render_statement_html(statement))
    assert r"$\{x \in \mathbb{R} : x > 0\}$" in decoded
    assert r"$a \cdot b = c$" in decoded


def test_matrices_and_ampersands_survive() -> None:
    """Las matrices (``&`` y ``\\\\``) no se escapan ni se colapsan."""
    statement = r"La matriz $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$."
    decoded = _decoded(render_statement_html(statement))
    assert r"$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$" in decoded


def test_display_math_is_preserved() -> None:
    """El bloque ``$$...$$`` se mantiene para que KaTeX lo renderice en display."""
    statement = r"$$ \int_0^1 x^2\,dx = \frac{1}{3} $$"
    decoded = _decoded(render_statement_html(statement))
    assert "$$" in decoded
    assert r"\int_0^1 x^2\,dx = \frac{1}{3}" in decoded


def test_markdown_formatting_still_works() -> None:
    """El Markdown del texto (negrita) sigue renderizándose."""
    rendered = render_statement_html("Texto **negrita** y $x_1$.")
    assert "<strong>negrita</strong>" in rendered


def test_no_sentinel_leaks() -> None:
    """Los centinelas internos nunca deben aparecer en la salida."""
    rendered = render_statement_html("$x_1$ y $y_2$ y texto sin fórmulas")
    assert "\ue000" not in rendered
    assert "\ue001" not in rendered


def test_empty_statement() -> None:
    """Un enunciado vacío produce HTML vacío."""
    assert render_statement_html("") == ""


def test_katex_assets_and_marker() -> None:
    """El CDN de KaTeX y la clase marcadora son los esperados."""
    assert MATH_CLASS == "taygeta-math"
    assert "katex" in KATEX_CDN
