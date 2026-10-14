"""Pruebas del padding de los recortes en ``image_extractor``.

Los recortes manuales de la UI deben ser exactos (sin padding), mientras que
los recortes automáticos del pipeline conservan un margen preventivo del 1 %
para compensar las BBox imprecisas del detector/LLM.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from taygeta.ingest.image_extractor import (
    DEFAULT_PADDING,
    MANUAL_CROP_PADDING,
    crop_figure_bytes,
    denormalize_bbox,
    extract_and_save_figures,
)

#: BBox de prueba ``[ymin, xmin, ymax, xmax]`` en una página de 1000x1000 px.
_BBOX = [100, 200, 400, 700]


def _source_png(tmp_path: Path, size: tuple[int, int] = (1000, 1000)) -> Path:
    """Crea una página PNG blanca para recortar sin depender de un PDF real."""
    source = tmp_path / "pagina.png"
    Image.new("RGB", size, "white").save(source)
    return source


def test_default_paddings() -> None:
    """El padding automático es del 1 % y el manual no añade margen."""
    assert DEFAULT_PADDING == pytest.approx(0.01)
    assert MANUAL_CROP_PADDING == 0.0


def test_manual_crop_matches_bbox_exactly(tmp_path: Path) -> None:
    """El recorte manual (padding 0) mide exactamente la extensión de la BBox."""
    png = crop_figure_bytes(_source_png(tmp_path), 0, _BBOX, padding=MANUAL_CROP_PADDING)
    with Image.open(io.BytesIO(png)) as crop:
        assert crop.size == (500, 300)


def test_manual_crop_is_tighter_than_padded_crop(tmp_path: Path) -> None:
    """Sin padding el recorte es más pequeño que el automático con margen."""
    source = _source_png(tmp_path)
    tight = crop_figure_bytes(source, 0, _BBOX, padding=MANUAL_CROP_PADDING)
    padded = crop_figure_bytes(source, 0, _BBOX, padding=DEFAULT_PADDING)
    with (
        Image.open(io.BytesIO(tight)) as tight_image,
        Image.open(io.BytesIO(padded)) as padded_image,
    ):
        assert tight_image.size == (500, 300)
        assert padded_image.size == (520, 320)


def test_denormalize_bbox_applies_one_percent_padding() -> None:
    """``denormalize_bbox`` añade el 1 % del tamaño de página en cada lado."""
    coords = denormalize_bbox(_BBOX, 1000.0, 1000.0, padding=DEFAULT_PADDING)
    assert coords == pytest.approx((190.0, 90.0, 710.0, 410.0))


def test_extract_and_save_figures_uses_one_percent_padding(tmp_path: Path) -> None:
    """La extracción automática conserva el margen preventivo del 1 %."""
    crops = extract_and_save_figures(_source_png(tmp_path), 0, [_BBOX], tmp_path / "out")
    assert len(crops) == 1
    assert crops[0].metadata["padding"] == pytest.approx(DEFAULT_PADDING)
    assert crops[0].path is not None
    with Image.open(crops[0].path) as crop:
        assert crop.size == (520, 320)
