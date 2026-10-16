"""Tests for manual figure editing in the HITL review.

Covers the pure helpers of ``taygeta.ui.figure_editing`` (pixel/BBox
conversion, placeholder cleanup and reconciliation), the full-page filter
of ``detect_figures`` and the new prompt restrictions (do not declare
``fig:N`` without an explicit diagram).
"""

from __future__ import annotations

import pymupdf as fitz
import pytest

from taygeta.agents.structurer import _REFINING_SYSTEM_PROMPT, _STRUCTURING_SYSTEM_PROMPT
from taygeta.ingest.image_extractor import detect_figures, placeholder_for
from taygeta.ingest.pipeline import IngestPipeline
from taygeta.ui.figure_editing import (
    append_figure_placeholder,
    bbox_to_pixel_box,
    build_new_figure,
    next_figure_id,
    pixel_box_to_normalized_bbox,
    reconcile_figures_with_statement,
    remove_figure,
    strip_figure_placeholder,
)


class _FakePage:
    """Minimal PyMuPDF page to test ``detect_figures`` without a real PDF."""

    def __init__(
        self,
        images: list[tuple[int, list[fitz.Rect]]] | None = None,
        drawings: list[fitz.Rect] | None = None,
        width: float = 612.0,
        height: float = 792.0,
    ) -> None:
        self.rect = fitz.Rect(0, 0, width, height)
        self.number = 0
        self._images = images or []
        self._drawings = drawings or []

    def get_images(self, full: bool = True) -> list[tuple[int, int]]:
        return [(xref, 0) for xref, _rects in self._images]

    def get_image_rects(self, xref: int) -> list[fitz.Rect]:
        for current_xref, rects in self._images:
            if current_xref == xref:
                return list(rects)
        return []

    def get_drawings(self) -> list[dict[str, fitz.Rect]]:
        return [{"rect": rect} for rect in self._drawings]


# ---------------------------------------------------------------------------
# Pixel <-> normalized BBox conversion
# ---------------------------------------------------------------------------


def test_pixel_box_to_normalized_bbox() -> None:
    box = {"left": 100, "top": 200, "width": 200, "height": 200}
    assert pixel_box_to_normalized_bbox(box, width=1000, height=800) == [250, 100, 500, 300]


def test_bbox_to_pixel_box_roundtrip() -> None:
    bbox = [250, 100, 500, 300]
    xl, xr, yt, yb = bbox_to_pixel_box(bbox, width=1000, height=800)
    assert (xl, xr, yt, yb) == (100, 300, 200, 400)


@pytest.mark.parametrize(
    ("box", "width", "height"),
    [
        ({"left": 0, "top": 0, "width": 0, "height": 50}, 1000, 800),
        ({"left": 0, "top": 0, "width": 50, "height": 0}, 1000, 800),
        ({"left": -5, "top": 0, "width": 50, "height": 50}, 1000, 800),
        ({"left": 0, "top": 0, "width": 1100, "height": 50}, 1000, 800),
    ],
)
def test_pixel_box_invalid_rejected(box: dict[str, int], width: int, height: int) -> None:
    with pytest.raises(ValueError):
        pixel_box_to_normalized_bbox(box, width, height)


# ---------------------------------------------------------------------------
# Placeholder cleanup and reconciliation
# ---------------------------------------------------------------------------


def test_strip_figure_placeholder() -> None:
    statement = "Según la figura {{fig:1}} y {{fig:1}} resuelve:"
    assert strip_figure_placeholder(statement, "fig:1") == "Según la figura  y  resuelve:"


def test_strip_figure_placeholder_keeps_others() -> None:
    statement = "Usa {{fig:1}} junto con {{fig:2}}."
    assert strip_figure_placeholder(statement, "fig:1") == "Usa  junto con {{fig:2}}."


def test_reconcile_figures_with_statement() -> None:
    figures = [
        {"id": "fig:1", "page": 0, "bbox": [100, 100, 400, 400]},
        {"id": "fig:2", "page": 0, "bbox": [500, 100, 700, 300]},
    ]
    effective, orphans, dangling = reconcile_figures_with_statement(
        figures, "Observa {{fig:1}} y responde."
    )
    assert [figure["id"] for figure in effective] == ["fig:1"]
    assert [figure["id"] for figure in orphans] == ["fig:2"]
    assert dangling == []


def test_reconcile_detects_orphan_placeholders() -> None:
    figures = [{"id": "fig:1", "page": 0, "bbox": [100, 100, 400, 400]}]
    effective, orphans, dangling = reconcile_figures_with_statement(
        figures, "Analiza {{fig:1}} y {{fig:2}}."
    )
    assert len(effective) == 1
    assert orphans == []
    assert dangling == ["fig:2"]


# ---------------------------------------------------------------------------
# Add figures manually (false negatives): sequential id and placeholder
# ---------------------------------------------------------------------------


def test_next_figure_id_starts_at_one() -> None:
    assert next_figure_id([], "") == "fig:1"


def test_next_figure_id_sequential_with_gaps() -> None:
    figures = [
        {"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1]},
        {"id": "fig:3", "page": 0, "bbox": [0, 0, 1, 1]},
    ]
    assert next_figure_id(figures, "Observa {{fig:1}}.") == "fig:4"


def test_next_figure_id_avoids_dangling_placeholders() -> None:
    figures = [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1]}]
    statement = "Analiza {{fig:1}} y {{fig:2}}."
    assert next_figure_id(figures, statement) == "fig:3"


def test_next_figure_id_ignores_non_numeric_ids() -> None:
    assert next_figure_id([{"id": "fig:abc", "page": 0, "bbox": [0, 0, 1, 1]}], "texto") == "fig:1"


def test_append_figure_placeholder() -> None:
    assert append_figure_placeholder("Resuelve.", "fig:2") == "Resuelve.\n{{fig:2}}"
    assert append_figure_placeholder("", "fig:1") == "{{fig:1}}"
    assert append_figure_placeholder("   ", "fig:1") == "{{fig:1}}"


# ---------------------------------------------------------------------------
# Adding and removing figures (pure helpers build_new_figure / remove_figure)
# ---------------------------------------------------------------------------


def test_build_new_figure_assigns_next_id_and_placeholder() -> None:
    statement = "Analiza {{fig:1}} y responde."
    figures = [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"}]
    nuevas, nuevo_enunciado, figure_id = build_new_figure(
        figures,
        statement,
        page_index=2,
        bbox=[100, 100, 400, 300],
        asset_path="/buckets/taygeta-assets/figures/nueva.png",
    )
    assert figure_id == "fig:2"
    assert nuevo_enunciado == "Analiza {{fig:1}} y responde.\n{{fig:2}}"
    assert [fig["id"] for fig in nuevas] == ["fig:1", "fig:2"]
    assert nuevas[-1] == {
        "id": "fig:2",
        "page": 2,
        "bbox": [100, 100, 400, 300],
        "caption": None,
        "asset_path": "/buckets/taygeta-assets/figures/nueva.png",
    }
    # The helper is pure: the original list is not mutated.
    assert len(figures) == 1


def test_build_new_figure_with_empty_list() -> None:
    nuevas, nuevo_enunciado, figure_id = build_new_figure(
        [],
        "",
        page_index=0,
        bbox=[1, 2, 3, 4],
        asset_path="k.png",
    )
    assert figure_id == "fig:1"
    assert nuevo_enunciado == "{{fig:1}}"
    assert len(nuevas) == 1


def test_remove_figure_removes_figure_and_placeholder() -> None:
    figures = [
        {"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"},
        {"id": "fig:2", "page": 1, "bbox": [0, 0, 1, 1], "asset_path": "b.png"},
    ]
    statement = "Observa {{fig:1}} y luego {{fig:2}}."
    kept, limpio, removed = remove_figure(figures, statement, "fig:1")
    assert [fig["id"] for fig in kept] == ["fig:2"]
    assert "{{fig:1}}" not in limpio
    assert "{{fig:2}}" in limpio
    assert removed == [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"}]
    # The original list is not mutated.
    assert len(figures) == 2


def test_remove_figure_nonexistent_changes_nothing() -> None:
    figures = [{"id": "fig:1", "page": 0, "bbox": [0, 0, 1, 1], "asset_path": "a.png"}]
    kept, limpio, removed = remove_figure(figures, "Con {{fig:1}}.", "fig:9")
    assert kept == figures
    assert limpio == "Con {{fig:1}}."
    assert removed == []


# ---------------------------------------------------------------------------
# Detection: full-page discard
# ---------------------------------------------------------------------------


def test_detect_figures_drops_full_page_scan() -> None:
    page = _FakePage(images=[(1, [fitz.Rect(0, 0, 612, 792)])])
    assert detect_figures(page) == []


def test_detect_figures_drops_near_full_page_and_keeps_small() -> None:
    full = fitz.Rect(0, 0, 500, 700)  # ≈ 72.2 % of the area -> discarded
    small = fitz.Rect(520, 50, 560, 150)  # ≈ 0.8 % -> kept
    page = _FakePage(images=[(1, [full]), (2, [small])])
    figures = detect_figures(page)
    assert [figure.id for figure in figures] == ["fig:1"]
    assert figures[0].bbox[0] == round(50 / 792 * 1000)  # ymin of the small rect


def test_detect_figures_keeps_regular_figure() -> None:
    page = _FakePage(images=[(1, [fitz.Rect(100, 100, 400, 300)])])
    figures = detect_figures(page)
    assert len(figures) == 1
    assert placeholder_for(figures[0].id) == "{{fig:1}}"


def test_detect_figures_ignores_rule_drawings() -> None:
    page = _FakePage(drawings=[fitz.Rect(0, 10, 612, 11)])  # horizontal rule
    assert detect_figures(page) == []


# ---------------------------------------------------------------------------
# Prompts: explicit figures only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [_STRUCTURING_SYSTEM_PROMPT, _REFINING_SYSTEM_PROMPT])
def test_prompts_require_explicit_figures(prompt: str) -> None:
    assert "do NOT declare fig:N" in prompt


def test_figure_hint_no_longer_orders_placeholders() -> None:
    from taygeta.core.schema import FigureReference

    hint = IngestPipeline._figure_hint(
        [FigureReference(id="fig:1", page=0, bbox=[100, 100, 500, 500])], page_index=0
    )
    assert "ONLY if the statement explicitly depends" in hint
    assert "do not insert any placeholder" in hint
