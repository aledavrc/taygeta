"""Previsualización del enunciado con fórmulas TeX (Markdown + KaTeX).

``ui.markdown`` se apoya en ``markdown2`` y no incluye ningún motor de
matemáticas; además ``markdown2`` altera las fórmulas antes de que ningún
renderizador pueda verlas (``$x_1$`` → ``$x<em>1</em>$``, ``\\{`` → ``{``,
``\\\\`` → ``\\``, ``&`` → ``&amp;``). Por eso el recuadro de formato usa
``ui.html`` con este renderizador, que:

1. extrae los fragmentos ``$$...$$`` / ``$...$`` y los sustituye por centinelas;
2. renderiza el Markdown del resto con ``markdown2``;
3. reinserta los fragmentos intactos (con escape HTML) dentro de ``$...$`` para
   que KaTeX los procese ya en el navegador.

KaTeX se carga una sola vez por página desde un CDN (:func:`install_math_rendering`)
y un ``MutationObserver`` vuelve a renderizar las fórmulas cada vez que el panel
actualiza la previsualización.
"""

from __future__ import annotations

import html
import re

import markdown2
from nicegui import ui

#: Versión de KaTeX servida desde el CDN.
KATEX_VERSION = "0.19.0"

#: URL base de los recursos de KaTeX.
KATEX_CDN = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"

#: Clase que marca el contenedor cuyas fórmulas debe procesar KaTeX.
MATH_CLASS = "taygeta-math"

#: Centinelas (área de uso privado de Unicode) que marcan dónde va cada fórmula.
_SENTINEL_OPEN = "\ue000"
_SENTINEL_CLOSE = "\ue001"

#: Fragmentos ``$$...$$`` (display) o ``$...$`` (inline). La alternativa ``\$\$``
#: va primero para que ``$$...$$`` no se parta en dos ``$...$``.
_MATH_RE = re.compile(r"(\$\$.+?\$\$|\$.+?\$)", re.DOTALL)

#: Extensiones de markdown2 equivalentes a las que usa ``ui.markdown``.
_MARKDOWN_EXTRAS = ["fenced-code-blocks", "tables"]

#: Cabecera que carga KaTeX y activa el re-renderizado incremental.
_KATEX_HEAD_HTML = f"""
<link rel="stylesheet" href="{KATEX_CDN}/katex.min.css" crossorigin="anonymous">
<script>
(function () {{
  var DELIMITERS = [
    {{left: "$$", right: "$$", display: true}},
    {{left: "\\\\[", right: "\\\\]", display: true}},
    {{left: "$", right: "$", display: false}},
    {{left: "\\\\(", right: "\\\\)", display: false}}
  ];
  var scheduled = false;
  function renderAll() {{
    scheduled = false;
    if (typeof window.renderMathInElement !== "function") return;
    document.querySelectorAll(".{MATH_CLASS}").forEach(function (el) {{
      if (el.querySelector(".katex")) return;
      try {{
        window.renderMathInElement(el, {{delimiters: DELIMITERS, throwOnError: false}});
      }} catch (err) {{ console.error("KaTeX:", err); }}
    }});
  }}
  function schedule() {{
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(renderAll);
  }}
  window.__taygetaRenderMath = schedule;
  document.addEventListener("DOMContentLoaded", schedule);
  new MutationObserver(schedule).observe(document.documentElement, {{
    childList: true, subtree: true, characterData: true
  }});
  schedule();
}})();
</script>
<script defer src="{KATEX_CDN}/katex.min.js" crossorigin="anonymous"
        onload="window.__taygetaRenderMath()"></script>
<script defer src="{KATEX_CDN}/contrib/auto-render.min.js" crossorigin="anonymous"
        onload="window.__taygetaRenderMath()"></script>
"""


def install_math_rendering() -> None:
    """Inyecta KaTeX en el ``<head>`` de la página actual.

    Debe llamarse dentro del constructor de una página (``@ui.page``); añade los
    recursos de KaTeX y el re-renderizado incremental de ``.taygeta-math``.
    """
    ui.add_head_html(_KATEX_HEAD_HTML)


def render_statement_html(text: str) -> str:
    """Renderiza el enunciado a HTML protegiendo las fórmulas TeX.

    Las fórmulas se extraen antes de pasar por ``markdown2`` y se reinsertan
    intactas (con el ``$...$`` original escapado para HTML) para que KaTeX las
    interprete en el navegador. El HTML resultante debe mostrarse dentro de un
    contenedor con la clase :data:`MATH_CLASS`.
    """
    if not text:
        return ""
    blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"{_SENTINEL_OPEN}{len(blocks) - 1}{_SENTINEL_CLOSE}"

    protected = _MATH_RE.sub(_stash, text)
    rendered = markdown2.markdown(protected, extras=_MARKDOWN_EXTRAS)
    for index, formula in enumerate(blocks):
        token = f"{_SENTINEL_OPEN}{index}{_SENTINEL_CLOSE}"
        rendered = rendered.replace(token, html.escape(formula, quote=False))
    return rendered
