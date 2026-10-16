"""Manual figure editing during the HITL review (pure helpers, no framework).

Centralizes the logic reused between the UI and the tests:

* conversion of the cropper's pixel box into the normalized BBox
  ``[ymin, xmin, ymax, xmax]`` (0-1000) required by ``FigureReference``;
* inverse conversion (normalized BBox -> pixels) to seed the cropper's initial
  box;
* removal of the ``{{fig:N}}`` placeholder from the statement when deleting a
  figure;
* reconciliation of figures/placeholders preserving the one-to-one mapping that
  ``QuestionRaw`` enforces when confirming the question.

This module imports no UI library so its functions are unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from taygeta.core.schema import PLACEHOLDER_PATTERN

__all__ = [
    "NORMALIZED_MAX",
    "append_figure_placeholder",
    "bbox_to_pixel_box",
    "build_new_figure",
    "next_figure_id",
    "pixel_box_to_normalized_bbox",
    "reconcile_figures_with_statement",
    "remove_figure",
    "strip_figure_placeholder",
]

#: Scale of the normalized BBox (same constant as ``image_extractor``).
NORMALIZED_MAX = 1000

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)

#: Captures the number of a canonical figure id (``fig:12`` -> 12).
_ID_NUMBER_RE = re.compile(r"^fig:(\d+)$")


def _figure_id(figure: Mapping[str, Any] | Any) -> str:
    """Return the ``id`` of a figure represented as a model or mapping."""
    if isinstance(figure, Mapping):
        return str(figure.get("id") or "")
    return str(getattr(figure, "id", "") or "")


def pixel_box_to_normalized_bbox(
    box: Mapping[str, int],
    width: int,
    height: int,
) -> list[int]:
    """Convert the cropper box (pixels) into a normalized 0-1000 BBox.

    ``box`` uses the keys ``left``/``top``/``width``/``height`` returned by
    ``st_cropper(return_type="box")`` over an image of ``width`` x ``height``
    pixels. Returns ``[ymin, xmin, ymax, xmax]`` with the same guarantees as the
    ``FigureReference`` validator (coordinates between 0 and 1000 and
    non-degenerate sides).

    Raises:
        ValueError: if the dimensions are not positive or the box is out of
            bounds or degenerate after normalization.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    left = int(box.get("left", 0))
    top = int(box.get("top", 0))
    box_width = int(box.get("width", 0))
    box_height = int(box.get("height", 0))
    if box_width <= 0 or box_height <= 0:
        raise ValueError("the crop box must have positive width and height")
    if left < 0 or top < 0 or left + box_width > width or top + box_height > height:
        raise ValueError("the crop box is out of the image bounds")

    xmin = round(left / width * NORMALIZED_MAX)
    ymin = round(top / height * NORMALIZED_MAX)
    xmax = round((left + box_width) / width * NORMALIZED_MAX)
    ymax = round((top + box_height) / height * NORMALIZED_MAX)

    if ymin >= ymax or xmin >= xmax:
        raise ValueError("the crop box is degenerate after normalization")

    return [
        min(max(ymin, 0), NORMALIZED_MAX),
        min(max(xmin, 0), NORMALIZED_MAX),
        min(max(ymax, 0), NORMALIZED_MAX),
        min(max(xmax, 0), NORMALIZED_MAX),
    ]


def bbox_to_pixel_box(
    normalized_bbox: Sequence[int | float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a normalized BBox into pixels: ``(xl, xr, yt, yb)``.

    The ``(xl, xr, yt, yb)`` format is what ``st_cropper`` expects in
    ``default_coords`` to seed the initial crop box.

    Raises:
        ValueError: if the dimensions are not positive or the BBox does not have
            4 coordinates.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(normalized_bbox) != 4:
        raise ValueError("the normalized BBox must have four coordinates")
    ymin, xmin, ymax, xmax = (float(value) for value in normalized_bbox)
    xl = round(xmin / NORMALIZED_MAX * width)
    xr = round(xmax / NORMALIZED_MAX * width)
    yt = round(ymin / NORMALIZED_MAX * height)
    yb = round(ymax / NORMALIZED_MAX * height)
    return xl, max(xr, xl + 1), yt, max(yb, yt + 1)


def strip_figure_placeholder(statement: str, figure_id: str) -> str:
    """Remove all ``{{fig:N}}`` placeholders of ``figure_id`` from the statement.

    Returns the statement without those placeholders (e.g. ``{{fig:1}}`` is
    removed when deleting figure ``fig:1``). If it does not appear, it is
    returned unchanged.
    """
    pattern = re.compile(r"\{\{" + re.escape(figure_id) + r"\}\}")
    return pattern.sub("", statement)


def next_figure_id(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
) -> str:
    """Return the next free sequential figure id.

    The number is computed from the set of ``fig:N`` ids declared in ``figures``
    **and** the ``{{fig:N}}`` placeholders present in the statement, so the new
    figure never collides with existing figures or with a placeholder (declared
    or dangling). If there are none, it returns ``fig:1``.
    """
    used: set[int] = set()
    for figure in figures:
        match = _ID_NUMBER_RE.match(_figure_id(figure))
        if match:
            used.add(int(match.group(1)))
    for token in _PLACEHOLDER_RE.findall(statement):
        used.add(int(token))
    return f"fig:{max(used, default=0) + 1}"


def append_figure_placeholder(statement: str, figure_id: str) -> str:
    """Insert the ``{{fig:X}}`` placeholder at the end of a statement.

    If the statement is empty (or only whitespace), only the placeholder is
    returned; otherwise it is appended on its own line, so it is easy to locate
    and move in the Markdown editor.
    """
    placeholder = "{{" + figure_id + "}}"
    if not statement or not statement.strip():
        return placeholder
    return f"{statement}\n{placeholder}"


def reconcile_figures_with_statement(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
) -> tuple[list[Any], list[Any], list[str]]:
    """Reconcile the figures with the placeholders of the edited statement.

    Returns a tuple ``(effective, orphans, dangling)``:

    * ``effective``: figures whose ``id`` appears as a placeholder in the
      statement (they count in the final question);
    * ``orphans``: declared figures that are no longer referenced (their asset
      should be deleted from the bucket);
    * ``dangling``: ``fig:N`` ids present in the statement with no declared
      figure (they would invalidate ``QuestionRaw`` and must block approval).

    The original order of ``figures`` is preserved in ``effective``; renumbering
    afterwards, if desired, is a decision of the review flow.
    """
    referenced: set[str] = {f"fig:{match}" for match in _PLACEHOLDER_RE.findall(statement)}
    declared: list[Any] = list(figures)

    dangling = sorted(referenced - {_figure_id(figure) for figure in declared})
    effective = [figure for figure in declared if _figure_id(figure) in referenced]
    orphans = [figure for figure in declared if _figure_id(figure) not in referenced]
    return effective, orphans, dangling


def _as_dict(figure: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert a figure (mapping or model) to a flat dictionary."""
    if isinstance(figure, Mapping):
        return dict(figure)
    return getattr(figure, "model_dump", lambda: {})()  # type: ignore[operator]


def build_new_figure(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
    *,
    page_index: int,
    bbox: Sequence[int],
    asset_path: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Build the entry of a new figure and the statement with its placeholder.

    Assigns the next free ``fig:N`` from ``figures`` and the placeholders of
    ``statement`` (see ``next_figure_id``) and returns
    ``(figures, statement, figure_id)``. It is a pure helper: cropping and
    uploading the asset to the bucket is the caller's responsibility, who
    already has ``asset_path`` at call time.
    """
    figure_id = next_figure_id(list(figures), statement)
    new_figure = {
        "id": figure_id,
        "page": int(page_index),
        "bbox": list(bbox),
        "caption": None,
        "asset_path": asset_path,
    }
    return [*figures, new_figure], append_figure_placeholder(statement, figure_id), figure_id


def remove_figure(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
    figure_id: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Remove a figure and its placeholder from the statement (no side effects).

    Returns ``(remaining_figures, clean_statement, removed)``. The caller
    decides whether to purge the removed figures' assets from the bucket.
    """
    kept = [_as_dict(item) for item in figures if _figure_id(item) != figure_id]
    removed = [_as_dict(item) for item in figures if _figure_id(item) == figure_id]
    return kept, strip_figure_placeholder(statement, figure_id), removed
