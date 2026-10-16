"""Tests for the TeX→Typst converter delegated to tylax.

Covers the replacement of the in-house parser by the :class:`MathConverter`
facade over ``tylax.latex_to_typst``: a golden corpus of representative
formulas, elevation to :class:`TranspileError` of tylax's syntax error marker,
passthrough of unknown commands and real compilation with the ``typst``
binary (skipped if not on PATH).
"""

from __future__ import annotations

import shutil

import pytest

from taygeta.core.transpiler import (
    MathConverter,
    TranspileError,
    compile_typst,
    markdown_to_typst,
    validate_typst,
)

#: ``(TeX, expected Typst output)`` pairs with the module's fixed options
#: (``frac_to_slash=False``, ``non_strict=True``).
_CORPUS: tuple[tuple[str, str], ...] = (
    (r"\frac{1}{2}", "frac(1, 2)"),
    (r"\frac{\partial f}{\partial x}", "frac(partial f, partial x)"),
    (r"\sqrt{x}", "sqrt(x)"),
    (r"\sqrt[3]{x}", "root(3, x)"),
    (r"\alpha\beta", "alpha beta"),
    (r"\Gamma(p+q)", "Gamma (p + q)"),
    (r"\mathbb{R}", "RR"),
    (r"\vec{v}", "arrow(v)"),
    (r"\bar{x}", "overline(x)"),
    (r"\sum_{i=1}^{n} i", "sum_(i = 1)^(n) i"),
    (r"\int_0^1 x\,dx", "integral_(0)^(1) x thin d x"),
    (r"\lim_{x\to 0} f(x)", "lim_(x -> 0) f (x)"),
    (r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", 'mat(delim: "(", a, b ; c, d)'),
    (
        r"\begin{cases} x & x>0 \\ -x & x\le 0 \end{cases}",
        "cases(x & x > 0, - x & x <= 0)",
    ),
    (r"\begin{aligned} x &= 1 \\ y &= 2 \end{aligned}", "x & = 1 \\ y & = 2"),
    (r"\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}", "mat(delim: #none, 1, 2 ; 3, 4)"),
    (r"\left| x \right|", "abs(x)"),
    (r"\left. \frac{1}{2} \right|_0^1", "lr(frac(1, 2) bar.v)_(0)^(1)"),
    (r"x \cdot y", "x dot y"),
    (r"\mid", "divides"),
    (r"\subseteq", "subset.eq"),
    (r"\notin", "in.not"),
    (r"\to", "->"),
    (r"x \text{ tal que } x>0", "x #text[tal que] x > 0"),
    (r"\mathrm{d}x", "dif x"),
    (r"\mathbf{A}", "upright(bold(A))"),
    (r"\hat{\theta}", "hat(theta)"),
    (r"\{ x \mid x>0 \}", "{ x divides x > 0 }"),
    (r"\emptyset", "emptyset"),
    (r"\varepsilon", "epsilon"),
    (r"\varphi", "phi"),
    (r"\ne", "!="),
    (r"\approx", "approx"),
)

_CORPUS_IDS = [tex for tex, _ in _CORPUS]


@pytest.mark.parametrize(("tex", "expected"), _CORPUS, ids=_CORPUS_IDS)
def test_golden_corpus_tex_to_typst(tex: str, expected: str) -> None:
    assert MathConverter().convert(tex) == expected


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst binary not available")
@pytest.mark.parametrize(("tex", "expected"), _CORPUS, ids=_CORPUS_IDS)
def test_corpus_compiles_with_real_typst(tex: str, expected: str, tmp_path) -> None:
    result = compile_typst(
        f"$ {expected} $\n", tmp_path, filename="main.typ", output_name="output.pdf"
    )
    assert result.ok, result.format_errors()


def test_unparseable_latex_raises_transpile_error() -> None:
    with pytest.raises(TranspileError) as exc_info:
        MathConverter().convert(r"}")
    assert exc_info.value.original == r"}"
    assert "tylax" in exc_info.value.message


def test_unknown_command_does_not_raise() -> None:
    # tylax is non-strict: it keeps the unknown command as raw text instead of
    # silently removing it; the final `typst compile` detects it.
    output = MathConverter().convert(r"\noop{foo}")
    assert "noop" in output


def test_markdown_to_typst_integrates_math_and_figures() -> None:
    output = markdown_to_typst(
        "Calcula $$\\int_0^1 x\\,dx$$ para $x > 0$.\n\n{{fig:1}}",
        figure_paths={"fig:1": "fig_1.png"},
    )
    assert "integral_(0)^(1)" in output
    assert "$x > 0$" in output
    assert '#figure(image("assets/fig_1.png"' in output


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst binary not available")
def test_validate_typst_accepts_generated_document() -> None:
    source = markdown_to_typst("Resuelve $$\\frac{1}{2}$$ para $x>0$.")
    result = validate_typst(source)
    assert result.ok, result.format_errors()
