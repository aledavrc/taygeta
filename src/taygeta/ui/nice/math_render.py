"""Statement preview with TeX formulas (Markdown + KaTeX).

``ui.markdown`` relies on ``markdown2`` and includes no math engine; moreover
``markdown2`` alters the formulas before any renderer can see them (``$x_1$``
→ ``$x<em>1</em>$``, ``\\{`` → ``{``, ``\\\\`` → ``\\``, ``&`` → ``&amp;``).
That is why the format box uses ``ui.html`` with this renderer, which:

1. extracts the ``$$...$$`` / ``$...$`` fragments and replaces them with
   sentinels;
2. renders the Markdown of the rest with ``markdown2``;
3. reinserts the fragments intact (HTML-escaped) inside ``$...$`` so that KaTeX
   processes them in the browser.

KaTeX is loaded once per page from a CDN (:func:`install_math_rendering`) and a
``MutationObserver`` re-renders the formulas every time the panel updates the
preview.
"""

from __future__ import annotations

import html
import re

import markdown2
from nicegui import ui

#: KaTeX version served from the CDN.
KATEX_VERSION = "0.19.0"

#: Base URL of the KaTeX resources.
KATEX_CDN = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"

#: Class marking the container whose formulas KaTeX must process.
MATH_CLASS = "taygeta-math"

#: Sentinels (Unicode private use area) marking where each formula goes.
_SENTINEL_OPEN = "\ue000"
_SENTINEL_CLOSE = "\ue001"

#: ``$$...$$`` (display) or ``$...$`` (inline) fragments. The ``\$\$``
#: alternative goes first so ``$$...$$`` is not split into two ``$...$``.
_MATH_RE = re.compile(r"(\$\$.+?\$\$|\$.+?\$)", re.DOTALL)

#: markdown2 extensions equivalent to the ones used by ``ui.markdown``.
_MARKDOWN_EXTRAS = ["fenced-code-blocks", "tables"]

#: Header that loads KaTeX and enables incremental re-rendering.
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
    """Inject KaTeX into the current page's ``<head>``.

    Must be called inside a page constructor (``@ui.page``); it adds the KaTeX
    resources and the incremental re-rendering of ``.taygeta-math``.
    """
    ui.add_head_html(_KATEX_HEAD_HTML)


def render_statement_html(text: str) -> str:
    """Render the statement to HTML protecting the TeX formulas.

    The formulas are extracted before passing through ``markdown2`` and
    reinserted intact (with the original ``$...$`` escaped for HTML) so KaTeX
    interprets them in the browser. The resulting HTML must be shown inside a
    container with the :data:`MATH_CLASS` class.
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
