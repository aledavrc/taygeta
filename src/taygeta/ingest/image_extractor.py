"""Preparation, rasterization, and figure geometry (section 4.1).

Responsibilities:

* Open PDF documents (and also PNG/JPG images) with PyMuPDF (``fitz``).
* Rasterize the pages at 300 DPI by default.
* Denormalize the 0-1000 scale BBoxes to the real page size by applying a
  preventive 1% padding on automatic crops; manual crops from the UI add no
  padding (exact crop).
* Keep the real zero-based physical page index and preserve the coordinates
  needed to associate each figure with the exact statement.
* Detect relevant figures and diagrams and assign ``{{fig:N}}`` placeholders
  in reading order.
* Create crops in a temporary working directory and, when requested, upload
  the final figures to SeaweedFS through ``object_storage``.

No absolute paths shared between containers are relied upon: crops are created
in local temporary directories and the final figures live in S3 under a stable
key.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pymupdf as fitz
from PIL import Image

from taygeta.core.schema import FigureReference
from taygeta.core.urls import relative_path_from_s3_uri

if TYPE_CHECKING:
    from taygeta.server.object_storage import ObjectStorage

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_PADDING",
    "MANUAL_CROP_PADDING",
    "MAX_AREA_RATIO",
    "NORMALIZED_MAX",
    "SUPPORTED_SUFFIXES",
    "CarriedContext",
    "FigureCrop",
    "ImageExtractionError",
    "PageRaster",
    "PixelRect",
    "assign_figure_ids",
    "carry_context",
    "denormalize_bbox",
    "detect_figures",
    "extract_and_save_figures",
    "crop_figure_bytes",
    "normalize_bbox",
    "open_document",
    "placeholder_for",
    "rasterize_document",
    "rasterize_page",
    "reading_order",
    "upload_crops",
    "working_directory",
]

#: Default rasterization resolution, required by the specification.
DEFAULT_DPI = 300

#: Units of the normalized BBox.
NORMALIZED_MAX = 1000

#: Default preventive padding for automatic crops (1% of the page size).
#: Compensates for the imprecise BBoxes from the detector/LLM.
DEFAULT_PADDING = 0.01

#: Padding for manual crops from the UI: no margin, fitted to the box.
MANUAL_CROP_PADDING = 0.0

#: Maximum fraction of the page area a figure may occupy. A candidate that
#: covers the whole page is, almost always, the document background or a scan
#: image, not a diagram: it is discarded so that the full sheet is not captured
#: as ``fig:N``.
MAX_AREA_RATIO = 0.6

#: Points per inch in the PDF coordinate space.
POINTS_PER_INCH = 72.0

#: Supported source extensions.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg"})


class ImageExtractionError(RuntimeError):
    """Controlled error during figure preparation or cropping."""


class PixelRect(NamedTuple):
    """Rectangle in pixel coordinates with the origin at the top-left corner."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(slots=True)
class PageRaster:
    """Page rasterized to a specific resolution."""

    page_index: int
    width_px: int
    height_px: int
    dpi: float
    png_bytes: bytes

    @property
    def size(self) -> tuple[int, int]:
        return (self.width_px, self.height_px)

    def save(self, path: str | Path) -> Path:
        """Save the rasterized page to disk and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.png_bytes)
        return destination


@dataclass(slots=True)
class FigureCrop:
    """Figure crop with its coordinates in several spaces."""

    figure_id: str
    page_index: int
    normalized_bbox: list[int]
    clip_points: tuple[float, float, float, float]
    pixel_rect: PixelRect
    dpi: float
    path: Path | None = None
    asset_uri: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CarriedContext:
    """Incomplete fragment from page N that serves as context for page N+1."""

    page_index: int
    text: str
    incomplete: bool = True
    placeholders: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Opening and rasterization
# ---------------------------------------------------------------------------


def is_supported_source(source: str | Path) -> bool:
    """Whether the extension of ``source`` is supported (PDF/PNG/JPG)."""
    return Path(source).suffix.lower() in SUPPORTED_SUFFIXES


@contextmanager
def open_document(source: str | Path) -> Iterator[fitz.Document]:
    """Open a PDF or image with PyMuPDF and guarantee that it is closed.

    Raises:
        ImageExtractionError: if the file does not exist, the extension is not
            supported, or PyMuPDF cannot open it.
    """
    path = Path(source)
    if not path.is_file():
        raise ImageExtractionError(f"source file does not exist: {path}")
    if not is_supported_source(path):
        raise ImageExtractionError(
            f"unsupported extension {path.suffix!r}; use one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    try:
        document = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises various types
        raise ImageExtractionError(f"could not open {path}: {exc}") from exc
    try:
        yield document
    finally:
        document.close()


def _document_is_pdf(document: fitz.Document) -> bool:
    return bool(getattr(document, "is_pdf", False))


def _image_native_size(path: str | Path) -> tuple[int, int] | None:
    """Return the native pixel size of an image file, if possible."""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - if it fails, the PyMuPDF metric is used
        return None


def rasterize_page(
    page: fitz.Page,
    *,
    dpi: int = DEFAULT_DPI,
    is_pdf: bool = True,
    zoom: float | None = None,
) -> PageRaster:
    """Rasterize a page and return its PNG bytes.

    For PDF, ``zoom = dpi / 72`` is used (300 DPI by default). For images, the
    ``zoom`` is calibrated so that the native pixel resolution is preserved.
    """
    if dpi <= 0:
        raise ImageExtractionError("DPI must be positive")
    if zoom is None:
        zoom = (dpi / POINTS_PER_INCH) if is_pdf else 1.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    except Exception as exc:
        raise ImageExtractionError(f"could not rasterize page {page.number}: {exc}") from exc
    return PageRaster(
        page_index=page.number,
        width_px=pixmap.width,
        height_px=pixmap.height,
        dpi=POINTS_PER_INCH * zoom,
        png_bytes=pixmap.tobytes("png"),
    )


def rasterize_document(source: str | Path, *, dpi: int = DEFAULT_DPI) -> list[PageRaster]:
    """Rasterize all pages of a document keeping the physical index."""
    with open_document(source) as document:
        is_pdf = _document_is_pdf(document)
        native = None if is_pdf else _image_native_size(source)
        pages = [
            rasterize_page(page, dpi=dpi, is_pdf=is_pdf, zoom=_page_zoom(page, is_pdf, dpi, native))
            for page in document
        ]
    logger.info("rasterized %d page(s) from %s", len(pages), source)
    return pages


def _page_zoom(
    page: fitz.Page,
    is_pdf: bool,
    dpi: int,
    native_size: tuple[int, int] | None,
) -> float:
    """Compute the scale factor to rasterize a page without losing detail."""
    if is_pdf:
        return dpi / POINTS_PER_INCH
    if native_size and page.rect.width > 0:
        return native_size[0] / page.rect.width
    return 1.0


# ---------------------------------------------------------------------------
# Geometry: normalization and denormalization
# ---------------------------------------------------------------------------


def denormalize_bbox(
    bbox: Sequence[int | float],
    page_width: float,
    page_height: float,
    *,
    padding: float = DEFAULT_PADDING,
    clamp: bool = True,
    as_int: bool = False,
) -> tuple[float, float, float, float]:
    """Convert a normalized 0-1000 BBox to real page coordinates.

    The BBox has the format ``[ymin, xmin, ymax, xmax]``. ``page_width`` and
    ``page_height`` are expressed in the target unit (page points). A preventive
    padding is applied (1% of the page size by default) on all four sides and,
    optionally, the result is clamped to the page area.

    Returns:
        Tuple ``(x0, y0, x1, y1)`` in the page unit.
    """
    if len(bbox) != 4:
        raise ImageExtractionError("BBox must have four values [ymin, xmin, ymax, xmax]")
    if page_width <= 0 or page_height <= 0:
        raise ImageExtractionError("page dimensions must be positive")
    if padding < 0:
        raise ImageExtractionError("padding cannot be negative")

    ymin, xmin, ymax, xmax = (float(value) for value in bbox)
    x0 = xmin / NORMALIZED_MAX * page_width
    x1 = xmax / NORMALIZED_MAX * page_width
    y0 = ymin / NORMALIZED_MAX * page_height
    y1 = ymax / NORMALIZED_MAX * page_height

    pad_x = padding * page_width
    pad_y = padding * page_height
    x0, x1 = x0 - pad_x, x1 + pad_x
    y0, y1 = y0 - pad_y, y1 + pad_y

    if clamp:
        x0 = min(max(x0, 0.0), page_width)
        x1 = min(max(x1, 0.0), page_width)
        y0 = min(max(y0, 0.0), page_height)
        y1 = min(max(y1, 0.0), page_height)

    if as_int:
        return (float(round(x0)), float(round(y0)), float(round(x1)), float(round(y1)))
    return (x0, y0, x1, y1)


def normalize_bbox(
    bbox: Sequence[int | float],
    page_width: float,
    page_height: float,
) -> list[int]:
    """Convert a BBox in page coordinates to the 0-1000 normalized scale."""
    if len(bbox) != 4:
        raise ImageExtractionError("BBox must have four values")
    if page_width <= 0 or page_height <= 0:
        raise ImageExtractionError("page dimensions must be positive")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    ymin = round(y0 / page_height * NORMALIZED_MAX)
    xmin = round(x0 / page_width * NORMALIZED_MAX)
    ymax = round(y1 / page_height * NORMALIZED_MAX)
    xmax = round(x1 / page_width * NORMALIZED_MAX)
    return [
        min(max(ymin, 0), NORMALIZED_MAX),
        min(max(xmin, 0), NORMALIZED_MAX),
        min(max(ymax, 0), NORMALIZED_MAX),
        min(max(xmax, 0), NORMALIZED_MAX),
    ]


def _rects_touch(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
    return not (a.x1 + gap < b.x0 or b.x1 + gap < a.x0 or a.y1 + gap < b.y0 or b.y1 + gap < a.y0)


def _merge_rects(rects: list[fitz.Rect], gap: float) -> list[fitz.Rect]:
    """Merge rectangles that overlap or are closer than ``gap`` to each other."""
    pending = list(rects)
    merged: list[fitz.Rect] = []
    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            remaining: list[fitz.Rect] = []
            for other in pending:
                if _rects_touch(current, other, gap):
                    current = current | other
                    changed = True
                else:
                    remaining.append(other)
            pending = remaining
        merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Figure detection and placeholders
# ---------------------------------------------------------------------------


def _is_rule_like(rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    """Whether a candidate looks like a decorative rule/line and not a figure.

    Underlines, inner table lines, or page edges are extremely thin and
    elongated rectangles that, when merged, pollute the detection: they are
    discarded when they occupy at least half of the page's larger dimension in
    one of its two directions.
    """
    if page_rect.width <= 0 or page_rect.height <= 0:
        return False
    horizontal = rect.height <= 2 and rect.width >= 0.5 * page_rect.width
    vertical = rect.width <= 2 and rect.height >= 0.5 * page_rect.height
    return horizontal or vertical


def detect_figures(
    page: fitz.Page,
    *,
    include_drawings: bool = True,
    min_area_ratio: float = 0.004,
    max_area_ratio: float = MAX_AREA_RATIO,
    max_figures: int = 50,
) -> list[FigureReference]:
    """Detect relevant figures on a page and return them normalized.

    Combines embedded images (``get_images``/``get_image_rects``) with vector
    drawings (``get_drawings``), discards tiny elements and decorative lines,
    and filters out candidates that cover almost the whole page
    (backgrounds/scans), so that a full sheet is never declared as ``fig:N``.
    Assigns ``fig:N`` identifiers in reading order.
    """
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1.0)
    candidates: list[fitz.Rect] = []

    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            if not rect.is_empty:
                candidates.append(rect)

    if include_drawings:
        for drawing in page.get_drawings():
            rect = fitz.Rect(drawing.get("rect", fitz.EMPTY_RECT))
            if rect.is_empty or _is_rule_like(rect, page_rect):
                continue
            candidates.append(rect)

    merged = _merge_rects(candidates, gap=2.0)
    relevant = [
        rect
        for rect in merged
        if min_area_ratio <= (rect.width * rect.height) / page_area <= max_area_ratio
        and rect.width > 1
        and rect.height > 1
    ]
    relevant.sort(key=lambda rect: (round(rect.y0, 1), round(rect.x0, 1)))
    relevant = relevant[:max_figures]

    figures: list[FigureReference] = []
    for index, rect in enumerate(relevant):
        normalized = normalize_bbox(
            (rect.x0, rect.y0, rect.x1, rect.y1), page_rect.width, page_rect.height
        )
        ymin, xmin, ymax, xmax = normalized
        if ymin >= ymax or xmin >= xmax:
            continue
        figures.append(
            FigureReference(
                id=f"fig:{index + 1}",
                page=page.number,
                bbox=normalized,
            )
        )
    return figures


def reading_order(figures: Sequence[FigureReference]) -> list[FigureReference]:
    """Sort figures by page and in reading flow (top->bottom, left->right)."""
    return sorted(
        figures,
        key=lambda figure: (
            figure.page,
            figure.bbox[0],
            figure.bbox[1],
        ),
    )


def assign_figure_ids(figures: Sequence[FigureReference]) -> list[FigureReference]:
    """Renumber the figures in reading order with ``fig:N`` identifiers.

    Keeps the rest of the fields (page, bbox, caption, asset_path) intact.
    """
    ordered = reading_order(figures)
    return [
        figure.model_copy(update={"id": f"fig:{index + 1}"}) for index, figure in enumerate(ordered)
    ]


def placeholder_for(figure_id: str) -> str:
    """Return the text placeholder associated with a figure identifier."""
    return "{{" + figure_id + "}}"


def carry_context(previous_text: str | None, page_index: int) -> CarriedContext | None:
    """Keep the incomplete fragment from page ``page_index`` as context.

    Returns ``None`` if there is no previous text. The fragment lets the OCR
    pass and the remote pass interpret exercises that continue on the next page.
    """
    if previous_text is None or not previous_text.strip():
        return None
    placeholders = [
        placeholder_for(f"fig:{token}") for token in _placeholder_numbers(previous_text)
    ]
    return CarriedContext(page_index=page_index, text=previous_text, placeholders=placeholders)


def _placeholder_numbers(text: str) -> list[str]:
    return re.findall(r"\{\{fig:(\d+)\}\}", text)


# ---------------------------------------------------------------------------
# Figure cropping
# ---------------------------------------------------------------------------


def _figure_spec(
    figure: FigureReference | Mapping[str, Any] | Sequence[int | float],
    index: int,
) -> tuple[str, list[int], str | None]:
    if isinstance(figure, FigureReference):
        return figure.id, list(figure.bbox), figure.caption
    if isinstance(figure, Mapping):
        figure_id = str(figure.get("id") or f"fig:{index + 1}")
        bbox = list(figure.get("bbox") or [])
        caption = figure.get("caption")
        return figure_id, [int(value) for value in bbox], caption
    bbox = [int(value) for value in figure]
    return f"fig:{index + 1}", bbox, None


def _safe_filename(figure_id: str, extension: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in figure_id)
    return f"{safe}{extension}"


def extract_and_save_figures(
    source: str | Path,
    page_index: int,
    figures: Sequence[FigureReference | Mapping[str, Any] | Sequence[int | float]],
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    padding: float = DEFAULT_PADDING,
    image_format: str = "png",
) -> list[FigureCrop]:
    """Crop and save the given figures into ``output_dir``.

    Denormalizes each 0-1000 BBox to the real page size adding the preventive
    padding, rasterizes the crop at ``dpi``, and writes the temporary file.
    Returns the normalized coordinates, the rectangle in points, and the
    rectangle in pixels to associate the figure with the exact statement.

    Args:
        source: Path of the PDF or image.
        page_index: Zero-based physical page index.
        figures: Figures with a normalized BBox (``FigureReference``, mapping, or
            sequence ``[ymin, xmin, ymax, xmax]``).
        output_dir: Temporary directory where the crops are written.
        dpi: Rasterization resolution of the crop.
        padding: Preventive padding as a fraction of the page size.
        image_format: Output image format (``png`` by default).
    """
    extension = f".{image_format.lstrip('.')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open_document(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise ImageExtractionError(
                f"page index {page_index} out of range (0..{document.page_count - 1})"
            )
        page = document.load_page(page_index)
        is_pdf = _document_is_pdf(document)
        page_rect = page.rect
        native = None if is_pdf else _image_native_size(source)
        zoom = _page_zoom(page, is_pdf, dpi, native)
        matrix = fitz.Matrix(zoom, zoom)
        effective_dpi = POINTS_PER_INCH * zoom

        crops: list[FigureCrop] = []
        for index, figure in enumerate(figures):
            figure_id, bbox, caption = _figure_spec(figure, index)
            x0, y0, x1, y1 = denormalize_bbox(
                bbox, page_rect.width, page_rect.height, padding=padding
            )
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                logger.warning("skipping figure %s: degenerate BBox %s", figure_id, bbox)
                continue
            clip = fitz.Rect(x0, y0, x1, y1)
            try:
                pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            except Exception as exc:
                raise ImageExtractionError(
                    f"could not crop figure {figure_id} on page {page_index}: {exc}"
                ) from exc

            destination = output_path / _safe_filename(figure_id, extension)
            pixmap.save(str(destination))
            pixel_rect = PixelRect(
                x0=round(x0 * zoom),
                y0=round(y0 * zoom),
                x1=round(x1 * zoom),
                y1=round(y1 * zoom),
            )
            crops.append(
                FigureCrop(
                    figure_id=figure_id,
                    page_index=page_index,
                    normalized_bbox=[int(round(value)) for value in bbox],
                    clip_points=(x0, y0, x1, y1),
                    pixel_rect=pixel_rect,
                    dpi=effective_dpi,
                    path=destination,
                    caption=caption,
                    metadata={
                        "padding": padding,
                        "page_width": page_rect.width,
                        "page_height": page_rect.height,
                    },
                )
            )
            logger.debug("figure %s cropped on page %d -> %s", figure_id, page_index, destination)
    return crops


def crop_figure_bytes(
    source: str | Path,
    page_index: int,
    bbox: Sequence[int],
    *,
    dpi: int = DEFAULT_DPI,
    padding: float = MANUAL_CROP_PADDING,
) -> bytes:
    """Crop a normalized region from a page and return its PNG bytes.

    Intended for manual crops from the UI: by default it adds **no** padding
    (``MANUAL_CROP_PADDING = 0``), so the PNG matches exactly the box selected
    by the user. It shares the denormalization and rasterization of
    :func:`extract_and_save_figures` (which does apply the preventive padding of
    automatic BBoxes), but returns the bytes in memory: this way the UI can
    overwrite an asset in S3 without touching the filesystem or leaving
    temporary copies.

    Raises:
        ImageExtractionError: if the page is out of range, the BBox is
            degenerate, or the crop fails.
    """
    with open_document(source) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise ImageExtractionError(
                f"page index {page_index} out of range (0..{document.page_count - 1})"
            )
        page = document.load_page(page_index)
        is_pdf = _document_is_pdf(document)
        page_rect = page.rect
        native = None if is_pdf else _image_native_size(source)
        zoom = _page_zoom(page, is_pdf, dpi, native)
        x0, y0, x1, y1 = denormalize_bbox(bbox, page_rect.width, page_rect.height, padding=padding)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            raise ImageExtractionError(f"degenerate BBox while cropping: {bbox}")
        try:
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(x0, y0, x1, y1), alpha=False
            )
        except Exception as exc:  # noqa: BLE001 - PyMuPDF raises various types
            raise ImageExtractionError(
                f"could not crop the figure from page {page_index}: {exc}"
            ) from exc
    return pixmap.tobytes("png")


def upload_crops(
    crops: Sequence[FigureCrop],
    storage: ObjectStorage,
    *,
    prefix: str = "figures",
    delete_local: bool = True,
    extension: str = ".png",
) -> list[FigureCrop]:
    """Upload the crops to SeaweedFS and record their stable relative path.

    The final figures live in S3 and are referenced through a relative path
    ``/buckets/{bucket}/{key}`` (the canonical storage form, independent of the
    endpoint). When ``delete_local`` is ``True`` the temporary copy is removed
    so a second local copy is not kept permanently.
    """
    for crop in crops:
        if crop.path is None:
            continue
        uri = storage.store_asset_file(crop.path, prefix=prefix, asset_id=None, extension=extension)
        crop.asset_uri = relative_path_from_s3_uri(uri)
        if delete_local:
            try:
                crop.path.unlink()
            except FileNotFoundError:
                pass
            crop.path = None
    return list(crops)


# ---------------------------------------------------------------------------
# Temporary working directory
# ---------------------------------------------------------------------------


@contextmanager
def working_directory(prefix: str = "taygeta-ingest-") -> Iterator[Path]:
    """Create a temporary working directory and remove it on exit.

    It does not touch the source uploaded by the user or the persistent objects.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
