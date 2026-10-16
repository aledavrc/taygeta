"""Helpers shared by Taygeta's HITL UI (framework-agnostic).

Centralizes the reusable UI logic:

* uploading/saving documents and listing the uploads directory;
* page rasterization and figure retrieval from SeaweedFS;
* composition of public URLs (internal/public S3 endpoints);
* building an :class:`EnrichedQuestion` from the reviewed fields and persisting
  approved questions (``persist_approved_questions``).

This module does not import NiceGUI, so it is unit-testable and reusable from
any layer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.schema import (
    EnrichedQuestion,
    FigureReference,
    QuestionRaw,
    get_public_url,
    normalize_slug,
    slugify,
    utc_now_iso,
)
from taygeta.core.settings import get_settings
from taygeta.core.urls import build_relative_path, split_asset_reference
from taygeta.i18n import t
from taygeta.ingest.image_extractor import (
    PageRaster,
    is_supported_source,
    open_document,
    rasterize_page,
)
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError
from taygeta.server.translation_service import TranslationService

logger = logging.getLogger(__name__)

__all__ = [
    "ConfirmResult",
    "build_enriched_question",
    "cache_busted_public_url",
    "delete_asset",
    "document_page_count",
    "figure_bytes",
    "list_uploads",
    "page_image_bytes",
    "page_image_raster",
    "persist_approved_questions",
    "public_asset_url",
    "public_s3_url",
    "sanitize_upload_name",
    "save_upload",
    "slug_split",
]

#: Local aliases of the shared helpers from ``taygeta.core.schema``.
_slugify = slugify
_normalize_slug = normalize_slug


def slug_split(text: str | None) -> list[str]:
    """Split a comma-separated string into normalized slugs (no empties)."""
    if not text:
        return []
    return [slug for slug in (_slugify(part) for part in text.split(",")) if slug]


# ---------------------------------------------------------------------------
# File and figure utilities (framework-testable)
# ---------------------------------------------------------------------------


def sanitize_upload_name(name: str) -> str:
    """Sanitize an uploaded file name, avoiding traversal and odd characters."""
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return cleaned or "upload"


def save_upload(name: str, data: bytes, uploads_dir: str | Path) -> Path:
    """Save the uploaded bytes into ``uploads_dir`` and return the final path."""
    directory = Path(uploads_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / sanitize_upload_name(name)
    if destination.exists():
        destination = directory / f"{destination.stem}_{uuid.uuid4().hex[:8]}{destination.suffix}"
    destination.write_bytes(data)
    return destination


def list_uploads(uploads_dir: str | Path) -> list[Path]:
    """List the supported documents present in ``uploads_dir``."""
    directory = Path(uploads_dir)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and is_supported_source(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def figure_bytes(storage: ObjectStorage | None, asset_path: str | None) -> bytes | None:
    """Retrieve a figure's bytes from SeaweedFS from its reference.

    Accepts ``/buckets/{bucket}/{key}``, ``s3://bucket/key`` or an object key.
    Returns ``None`` if the reference is empty or the object cannot be retrieved.
    """
    if storage is None or not asset_path:
        return None
    try:
        bucket, key = split_asset_reference(asset_path)
        if bucket:
            return storage.get_bytes(key, bucket=bucket)
        return storage.get_bytes(key)
    except (ObjectStorageError, ValueError) as exc:
        logger.warning("could not retrieve figure %s: %s", asset_path, exc)
        return None


def page_image_bytes(source_path: str | Path, page_index: int, *, dpi: int = 150) -> bytes | None:
    """Rasterize a page of the source document to show it during review."""
    try:
        with open_document(source_path) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            is_pdf = bool(getattr(document, "is_pdf", False))
            return rasterize_page(page, dpi=dpi, is_pdf=is_pdf).png_bytes
    except Exception as exc:  # noqa: BLE001 - the preview must never break the UI
        logger.warning("could not rasterize page %s: %s", page_index, exc)
        return None


def page_image_raster(
    source_path: str | Path, page_index: int, *, dpi: int = 150
) -> PageRaster | None:
    """Return a page's ``PageRaster`` for the cropper (bytes + size).

    ``PageRaster`` exposes ``png_bytes`` and ``width_px``/``height_px``, which
    are needed both to feed the manual crop and to convert the pixel box into
    the normalized BBox.
    """
    try:
        with open_document(source_path) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            is_pdf = bool(getattr(document, "is_pdf", False))
            return rasterize_page(page, dpi=dpi, is_pdf=is_pdf)
    except Exception as exc:  # noqa: BLE001 - the preview must never break the UI
        logger.warning("could not rasterize page %s: %s", page_index, exc)
        return None


def document_page_count(source_path: str | Path | None) -> int | None:
    """Return the page count of the source document, or ``None`` if unreachable.

    Used to offer the page selector for manually cropping figures.
    """
    if source_path is None:
        return None
    try:
        with open_document(source_path) as document:
            return document.page_count
    except Exception as exc:  # noqa: BLE001 - the UI must never break because of the source
        logger.warning("could not count pages of %s: %s", source_path, exc)
        return None


def delete_asset(storage: ObjectStorage | None, asset_path: str) -> None:
    """Delete an asset by its stable reference (best-effort, never blocks)."""
    if storage is None:
        return
    try:
        bucket, key = split_asset_reference(asset_path, default_bucket=storage.bucket)
        storage.delete(key, bucket=bucket or None)
    except (ObjectStorageError, ValueError) as exc:
        logger.warning("could not purge orphan asset %s: %s", asset_path, exc)


def public_s3_url(url: str | None, settings: Any) -> str:
    """Rewrite the internal S3 endpoint with the UI's public one.

    Replaces the ``S3_ENDPOINT_URL`` prefix (e.g. ``http://seaweedfs-s3:8333``)
    with ``S3_PUBLIC_ENDPOINT`` (e.g. ``http://localhost:8333``) so links and
    images can be opened from the host browser. If there is no public endpoint
    or the URL does not use the internal one, it is returned unchanged.
    """
    if not url:
        return ""
    internal = (settings.S3_ENDPOINT_URL or "").rstrip("/")
    public = (settings.S3_PUBLIC_ENDPOINT or "").rstrip("/")
    if public and internal and url.startswith(internal):
        return public + url[len(internal) :]
    return url


def public_asset_url(asset_path: str | None, settings: Any) -> str:
    """Compose a figure's public URL with the schema's ``get_public_url``.

    Converts any asset reference (``/buckets/{bucket}/{key}``,
    ``s3://bucket/key`` or a relative key) into its canonical relative path and
    concatenates it with ``S3_PUBLIC_BASE_URL``. This way the UI renders images
    with the host's public URL, never with internal Docker URLs (e.g.
    ``http://seaweedfs-s3:8333``).
    """
    if not asset_path:
        return ""
    try:
        bucket, key = split_asset_reference(asset_path, default_bucket=settings.S3_BUCKET)
        relative = build_relative_path(bucket, key)
    except ValueError:
        return ""
    return get_public_url(relative)


def cache_busted_public_url(asset_path: str | None, settings: Any, seed: Any = "") -> str:
    """Public URL of a figure with an anti-cache version parameter.

    The preview reads the bytes directly from SeaweedFS (no browser cache); the
    external link appends ``?v=`` derived from the asset and the BBox so the
    browser does not serve a stale version after a re-crop.
    """
    url = public_asset_url(asset_path, settings)
    if not url:
        return ""
    token = hashlib.sha256(f"{asset_path}|{seed}".encode()).hexdigest()[:8]
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={token}"


# ---------------------------------------------------------------------------
# Building and persisting approved questions
# ---------------------------------------------------------------------------


def build_enriched_question(
    base: QuestionRaw,
    *,
    course: str,
    topic: str,
    subtopic: str | None,
    difficulty: int,
    prerequisites: Sequence[str],
    source_metadata: dict[str, Any] | None = None,
    figures: Sequence[FigureReference] | None = None,
) -> EnrichedQuestion:
    """Build an :class:`EnrichedQuestion` from the reviewed fields.

    Manually edited values take precedence over the automatic transcription.
    ``course``, ``topic`` and ``subtopic`` are normalized to valid slugs
    (``^[a-z0-9_]+$``) to avoid ``ValidationError``; if ``course`` or ``topic``
    are left empty, ``"general"`` is used, and an empty ``subtopic`` becomes
    ``None``. Prerequisites are normalized the same way and empty ones are
    dropped. ``figures`` allows replacing ``base.figures`` with the selection
    edited during review (deletions and re-crops); if omitted, the originals are
    kept.
    """
    return EnrichedQuestion(
        id=base.id,
        statement_raw=base.statement_raw,
        question_type=base.question_type,
        choices=base.choices,
        figures=figures if figures is not None else base.figures,
        notation_notes=base.notation_notes,
        course=_normalize_slug(course),
        topic=_normalize_slug(topic),
        subtopic=_slugify(subtopic) or None,
        difficulty=difficulty,
        prerequisites=[slug for slug in (_slugify(item) for item in prerequisites) if slug],
        source_metadata=source_metadata or {},
        created_at=utc_now_iso(),
    )


@dataclass(slots=True)
class ConfirmResult:
    """Result of confirming and persisting questions."""

    saved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_figure_assets(question: EnrichedQuestion, storage: ObjectStorage | None) -> list[str]:
    missing: list[str] = []
    for figure in question.figures:
        if not figure.asset_path:
            missing.append(t("ui.persist.figure_missing_asset", figure_id=figure.id))
            continue
        if storage is None:
            continue
        try:
            bucket, key = split_asset_reference(figure.asset_path)
            if bucket:
                if not storage.exists(key, bucket=bucket):
                    missing.append(figure.asset_path)
            elif not storage.exists(key):
                missing.append(figure.asset_path)
        except (ObjectStorageError, ValueError):
            missing.append(figure.asset_path)
    return missing


def persist_approved_questions(
    questions: Sequence[EnrichedQuestion],
    database: Database,
    storage: ObjectStorage | None = None,
    *,
    session_id: str | None = None,
    embedder: Embedder | None = None,
    translations: TranslationService | None = None,
) -> ConfirmResult:
    """Persist the approved questions after verifying their figures in the bucket.

    Uses ``upsert_question`` so confirmation is idempotent. It only marks the
    staging session as ``CONFIRMED`` when it has saved at least one question.
    Generates (best-effort) each question's embedding for later semantic
    deduplication; if the provider is unavailable, it is skipped without
    blocking confirmation. With ``translations`` it also generates and stores
    the es/en copies, reporting any problem as a warning.
    """
    if embedder is None:
        embedder = Embedder(get_settings())
    result = ConfirmResult()
    for question in questions:
        try:
            missing = _missing_figure_assets(question, storage)
            if missing:
                result.errors.append(
                    t(
                        "ui.persist.figure_not_in_bucket",
                        question_id=question.id,
                        figures=", ".join(missing),
                    )
                )
                continue
            database.add_prerequisites(list(question.prerequisites))
            database.upsert_question(question, embedding=_embed_or_none(embedder, question))
            result.saved.append(question.id)
            if translations is not None:
                result.warnings.extend(translations.store(question).problems)
        except Exception as exc:  # noqa: BLE001 - reported to the user, not hidden
            result.errors.append(f"{question.id}: {exc}")

    if session_id and result.saved:
        try:
            database.update_staging_session(session_id, status="CONFIRMED")
        except DatabaseError as exc:
            result.errors.append(t("ui.persist.session_update_failed", error=exc))
    return result


def _embed_or_none(embedder: Embedder, question: EnrichedQuestion) -> list[float] | None:
    """Generate a question's embedding without raising on provider failures."""
    try:
        return embedder.embed(question_embedding_text(question))
    except Exception:  # noqa: BLE001 - semantic deduplication is optional
        logger.warning("could not generate the embedding for question %s", question.id)
        return None
