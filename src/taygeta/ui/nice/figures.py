"""Figure operations and components for the NiceGUI review.

``DragBoxCropper`` draws the crop box by dragging over
``ui.interactive_image`` (SVG overlay in image pixel coordinates, with no npm
dependencies), delivers the box in original image pixels and offers zoom
(1x-4x) to see precisely what is going to be cropped; the functions
``crop_new_figure`` and ``replace_figure_crop`` crop, upload and register
figures reusing the pure helpers of ``figure_editing`` and the shared
rasterization/storage clients.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nicegui import ui
from PIL import Image

from taygeta.core.urls import split_asset_reference
from taygeta.i18n import t
from taygeta.ingest.image_extractor import DEFAULT_DPI, crop_figure_bytes
from taygeta.server.object_storage import ObjectStorage
from taygeta.ui.figure_editing import build_new_figure
from taygeta.ui.nice import caption as _caption
from taygeta.ui.shared import figure_bytes, page_image_raster


def figure_pil(storage: ObjectStorage | None, asset_path: str | None) -> Image.Image | None:
    """Open a SeaweedFS figure as a ``PIL.Image`` for ``ui.image``."""
    data = figure_bytes(storage, asset_path)
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except OSError:
        return None


def page_pil(
    source_path: str | Path | None, page_index: int, *, dpi: int = 300
) -> tuple[Image.Image, int, int] | None:
    """Rasterize page ``page_index`` as a PIL image + size in pixels.

    Uses 300 DPI (same as the crops) so the cropper's zoom looks sharp on PDF
    documents; on scans the rasterization keeps the native resolution. Returns
    ``(image, width_px, height_px)`` or ``None`` if the source is unavailable or
    the page does not exist.
    """
    if source_path is None:
        return None
    raster = page_image_raster(source_path, int(page_index), dpi=dpi)
    if raster is None:
        return None
    image = Image.open(io.BytesIO(raster.png_bytes))
    image.load()
    return image, int(raster.width_px), int(raster.height_px)


def crop_new_figure(
    figures: Sequence[Mapping[str, Any] | Any],
    statement: str,
    *,
    page_index: int,
    bbox: Sequence[int],
    source_path: str | Path | None,
    storage: ObjectStorage | None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Crop, upload and register a new figure; returns (figures, statement, id).

    Covers false negatives: crops the indicated region of the original page
    (300 DPI), persists it with the pipeline's convention
    (``figures/{uuid}.png``) and delegates to ``build_new_figure`` the assignment
    of the next free ``fig:N`` and the insertion of its placeholder at the end of
    the statement.

    Raises:
        ValueError: if no source or storage is available.
    """
    if source_path is None:
        raise ValueError("no source document available to crop")
    if storage is None:
        raise ValueError("storage not available to store the figure")
    png = crop_figure_bytes(source_path, int(page_index), list(bbox), dpi=DEFAULT_DPI)
    uri = storage.store_asset_bytes(
        png,
        prefix="figures",
        asset_id=None,
        extension=".png",
        content_type="image/png",
    )
    asset_path = storage.relative_path(uri)
    return build_new_figure(
        figures,
        statement,
        page_index=int(page_index),
        bbox=list(bbox),
        asset_path=asset_path,
    )


def replace_figure_crop(
    figures: Sequence[Mapping[str, Any] | Any],
    figure_id: str,
    new_bbox: Sequence[int],
    source_path: str | Path | None,
    storage: ObjectStorage | None,
) -> list[dict[str, Any]]:
    """Re-crop a figure and overwrite its asset (same key) in SeaweedFS.

    Crops the indicated region of the source document at 300 DPI and overwrites
    the existing object (same key), so ``asset_path`` does not change. Returns
    the updated list with the new BBox.

    Raises:
        ValueError: if the figure has no persistent asset or services are missing.
    """
    if source_path is None:
        raise ValueError("no source document available to crop")
    if storage is None:
        raise ValueError("storage not available to store the crop")
    updated: list[dict[str, Any]] = []
    for item in figures:
        if isinstance(item, Mapping):
            figure = dict(item)
        else:
            figure = item.model_dump(mode="json")  # type: ignore[union-attr]
        if figure.get("id") != figure_id:
            updated.append(figure)
            continue
        asset_path = figure.get("asset_path")
        if not asset_path:
            raise ValueError(f"figure {figure_id} has no persistent asset; cannot overwrite")
        bucket, key = split_asset_reference(asset_path, default_bucket=storage.bucket)
        png = crop_figure_bytes(
            source_path, int(figure.get("page", 0)), list(new_bbox), dpi=DEFAULT_DPI
        )
        storage.put_bytes(key, png, content_type="image/png", bucket=bucket or None)
        updated.append({**figure, "bbox": list(new_bbox)})
    return updated


class DragBoxCropper:
    """Crop box drawn by dragging over an interactive image.

    Captures ``mousedown``/``mousemove``/``mouseup`` over
    ``ui.interactive_image`` and draws the rectangle as an SVG overlay (the
    ``content`` property, whose coordinates match the image pixels). On mouse
    release, it invokes ``on_confirm`` with ``{"left", "top", "width",
    "height"}`` in original image pixels (clamped to its dimensions).

    The confirmed box stays drawn after releasing the mouse (it is stored in
    ``self.box``), so the selection does not disappear. ``initial_box`` allows
    seeding the cropper with a figure's current region.

    Includes zoom controls (1x-4x): the image is resized by *layout* inside a
    scrollable container (never with ``transform: scale()``), so NiceGUI's
    mapping ``offsetX * naturalWidth / clientWidth`` and the SVG ``viewBox``
    remain exact at any magnification.
    """

    MIN_ZOOM = 1.0
    MAX_ZOOM = 4.0
    ZOOM_STEP = 0.5

    def __init__(
        self,
        source: Any,
        *,
        width: int,
        height: int,
        on_confirm: Callable[[dict[str, int]], None],
        caption: str | None = None,
        initial_box: dict[str, int] | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.on_confirm = on_confirm
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        #: Last confirmed box (or the initial seed); redrawn on release.
        self.box: dict[str, int] | None = initial_box
        self.zoom = self.MIN_ZOOM
        _caption(caption if caption is not None else t("ui.cropper.select_region"))

        with ui.row().classes("items-center gap-2"):
            self._zoom_out = (
                ui.button(
                    icon="remove",
                    on_click=lambda: self._set_zoom(self.zoom - self.ZOOM_STEP),
                )
                .props("dense round flat")
                .mark("cropper-zoom-out")
            )
            self._zoom_label = ui.label("100 %").classes("w-14 text-center text-sm")
            self._zoom_in = (
                ui.button(
                    icon="add",
                    on_click=lambda: self._set_zoom(self.zoom + self.ZOOM_STEP),
                )
                .props("dense round flat")
                .mark("cropper-zoom-in")
            )
            (
                ui.button(icon="restart_alt", on_click=lambda: self._set_zoom(self.MIN_ZOOM))
                .props("dense round flat")
                .mark("cropper-zoom-reset")
                .tooltip(t("ui.cropper.zoom_reset"))
            )

        # The border frames the viewport (not the image): when zooming in, the
        # image scrolls inside with the scrollbars.
        self._viewport = (
            ui.element("div")
            .classes("w-full overflow-auto border border-gray-300 rounded")
            .style("max-height: 75vh")
        )
        with self._viewport:
            self._canvas = ui.element("div").style("width: 100%")
            with self._canvas:
                self.image = ui.interactive_image(
                    source,
                    on_mouse=self._on_mouse,
                    events=["mousedown", "mousemove", "mouseup"],
                    cross=True,
                )
        self._refresh()
        self._apply_zoom()

    def _set_zoom(self, value: float) -> None:
        """Set the zoom to the allowed range; the controls use this entry point."""
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, round(value, 2)))
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        """Apply the zoom by *layout* width and sync the controls."""
        self._canvas.style(f"width: {int(round(self.zoom * 100))}%")
        self._zoom_label.set_text(f"{int(round(self.zoom * 100))} %")
        self._zoom_out.set_enabled(self.zoom > self.MIN_ZOOM)
        self._zoom_in.set_enabled(self.zoom < self.MAX_ZOOM)

    def _clamped(self, x: float, y: float) -> tuple[int, int]:
        return max(0, min(int(x), self.width)), max(0, min(int(y), self.height))

    def _on_mouse(self, e: Any) -> None:
        if e.type == "mousedown":
            self.start = self._clamped(e.image_x, e.image_y)
            self.current = self.start
            # Starting a new drag discards the previous box.
            self.box = None
        elif e.type == "mousemove" and self.start is not None:
            self.current = self._clamped(e.image_x, e.image_y)
        elif e.type == "mouseup" and self.start is not None:
            box = self._box()
            self.start = None
            self.current = None
            if box is not None:
                self.box = box
                self.on_confirm(box)
        self._refresh()

    def _box(self) -> dict[str, int] | None:
        if self.start is None or self.current is None:
            return None
        x0, y0 = self.start
        x1, y1 = self.current
        left, top = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width < 2 or height < 2:
            return None
        return {"left": left, "top": top, "width": width, "height": height}

    def _refresh(self) -> None:
        """Redraw the overlay with the in-progress box or the last confirmed one.

        After ``mouseup`` ``self.box`` is kept so the selection stays visible; a
        new ``mousedown`` clears it.
        """
        box = self._box() or self.box
        svg = ""
        if box is not None:
            svg = (
                f'<rect x="{box["left"]}" y="{box["top"]}" width="{box["width"]}" '
                f'height="{box["height"]}" fill="rgba(59,130,246,0.25)" '
                'stroke="#3b82f6" stroke-width="2"/>'
            )
        self.image.content = svg
        self.image.update()
