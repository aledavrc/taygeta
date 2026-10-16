"""Taygeta MCP server (section 9).

Exposes the FastMCP tools of the HITL flow and RAG retrieval:

* ``stage_exam_upload`` (headless ingestion with ``auto_commit``),
  ``refine_staged_question``, ``confirm_staged_upload`` (chat-assisted staging);
* ``search_questions`` and ``list_taxonomy`` (RAG retrieval and catalog);
* ``generate_practice_exam`` and ``get_exam_source`` (generation and compilation).

It also mounts the HTTP route ``/download/{filename}`` (and a controlled
``/assets/{asset_id}``) plus a background cleaner with a 60-minute TTL for the
temporary files in ``/app/downloads``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    QuestionRaw,
    TranscriptionIntermediate,
    sanitize_math_delimiters,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import build_public_url, build_relative_path, split_asset_reference
from taygeta.domain.taxonomy import TaxonomyNames, filter_taxonomy_names
from taygeta.i18n import is_supported_locale
from taygeta.ingest.pipeline import IngestPipeline, StagingResult
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database
from taygeta.server.deduplication import DeduplicationService
from taygeta.server.exam_generator import ExamGenerationError, ExamGenerator
from taygeta.server.object_storage import ObjectStorage, ObjectStorageError
from taygeta.server.translation_service import TranslationService

logger = logging.getLogger(__name__)

__all__ = ["mcp", "main", "start_download_cleaner"]

#: Editable fields of a question that accept manual overrides.
_ALLOWED_OVERRIDES = frozenset({"statement_raw", "question_type", "choices", "notation_notes"})

#: Curricular fields attached to the staging payload for confirmation.
_CURRICULAR_OVERRIDES = frozenset({"course", "topic", "subtopic", "difficulty", "prerequisites"})

#: Maximum search result limit to protect the service.
_MAX_SEARCH_LIMIT = 100

_settings = get_settings()
mcp = FastMCP("taygeta", host=_settings.MCP_HOST, port=_settings.MCP_PORT)


def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline, ExamGenerator]:
    settings = get_settings()
    database = Database(settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    exams = ExamGenerator(settings, database=database, storage=storage)
    return settings, database, storage, pipeline, exams


# ---------------------------------------------------------------------------
# Security and formatting utilities
# ---------------------------------------------------------------------------


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _public_question(
    question: EnrichedQuestion,
    settings: Settings,
    database: Database | None = None,
    *,
    locale: str | None = None,
) -> dict[str, Any]:
    """Serializes a question, resolving figures and attaching its translations.

    Before sending the information to the frontend or the MCP client, each
    relative figure path (``/buckets/{bucket}/{key}``) is turned into its full URL
    by concatenating ``S3_PUBLIC_BASE_URL``. With ``database`` the localized
    copies are attached: by default ``statement_by_locale`` (all locales) and, if
    ``locale`` is requested, also the resolved ``statement``.
    """
    data = question.model_dump()
    for figure in data.get("figures", []):
        reference = figure.get("asset_path")
        if reference:
            bucket, key = split_asset_reference(reference, default_bucket=settings.S3_BUCKET)
            figure["asset_path"] = build_public_url(build_relative_path(bucket, key), settings)
    if database is not None:
        by_locale = {
            entry.locale: entry.statement_raw
            for entry in database.list_question_translations(question.id)
        }
        data["statement_by_locale"] = by_locale
        if is_supported_locale(locale):
            data["locale"] = locale
            data["statement"] = by_locale.get(locale, question.statement_raw)
    return data


def _resolve_upload_path(file_path: str, settings: Settings) -> Path:
    """Validates that ``file_path`` is inside ``UPLOADS_DIR`` and is a file."""
    uploads = Path(settings.UPLOADS_DIR).expanduser().resolve()
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = uploads / candidate
    candidate = candidate.resolve()
    if candidate != uploads and not candidate.is_relative_to(uploads):
        raise ValueError("the path is outside the uploads directory")
    if not candidate.is_file():
        raise ValueError(f"the file does not exist: {candidate}")
    return candidate


def _resolve_download_path(filename: str, settings: Settings) -> Path:
    """Validates a download file name to prevent traversal."""
    downloads = Path(settings.DOWNLOADS_DIR).expanduser().resolve()
    if not filename or filename.startswith("/") or ".." in Path(filename).parts:
        raise ValueError("invalid file name")
    candidate = (downloads / filename).resolve()
    if not candidate.is_relative_to(downloads):
        raise ValueError("the download path is outside the allowed directory")
    if not candidate.is_file():
        raise ValueError("file not found")
    return candidate


def _apply_overrides(question: QuestionRaw, overrides: dict[str, Any] | None) -> QuestionRaw:
    """Applies only the allowed manual overrides on top of the question.

    ``model_copy`` does not run the Pydantic validators, so the manual statement
    and choices are explicitly sanitized here to avoid reintroducing non-standard
    TeX delimiters (``\\[...\\]``, ``\\(...\\)``).
    """
    if not overrides:
        return question
    updates: dict[str, Any] = {
        key: value for key, value in overrides.items() if key in _ALLOWED_OVERRIDES
    }
    if isinstance(updates.get("statement_raw"), str):
        updates["statement_raw"] = sanitize_math_delimiters(updates["statement_raw"])
    if "choices" in updates and updates["choices"] is not None:
        updates["choices"] = [Choice.model_validate(choice) for choice in updates["choices"]]
    return question.model_copy(update=updates)


# ---------------------------------------------------------------------------
# MCP tools — chat-assisted staging
# ---------------------------------------------------------------------------


@mcp.tool()
def stage_exam_upload(file_path: str, auto_commit: bool = True) -> str:
    """Processes a file from /app/uploads with the headless flow.

    Runs the two-pass pipeline and the two validation layers. If all questions
    are ``VALID`` and ``auto_commit`` is active, it stores them directly in the
    database and marks the session as ``CONFIRMED``. If there are issues
    (``NEEDS_ATTENTION``, ``ERROR`` or ``DUPLICATE``), it persists the staging
    session in the database and returns its identifier and the review URL, so
    that ``refine_staged_question`` and ``confirm_staged_upload`` close the loop.
    """
    try:
        settings, database, storage, pipeline, _exams = _resources()
        path = _resolve_upload_path(file_path, settings)
        result = pipeline.ingest(path, persist_staging=True)
    except Exception as exc:  # noqa: BLE001 - reported to the chat, never hidden
        return _json({"status": "error", "message": str(exc)})

    session_id = result.session_id
    if session_id is None:
        return _json(
            {
                "status": "error",
                "message": "could not persist the staging session in the database",
            }
        )

    summary = result.summary()
    problems = [
        {
            "id": item.question.id if item.question is not None else None,
            "status": item.status,
            "reasons": item.reasons,
        }
        for item in result.questions
        if item.status != STATUS_VALID or item.question is None
    ]

    extra_warnings: list[str] = []
    if auto_commit and result.status == STATUS_VALID and not problems:
        saved, save_problems, extra_warnings = _save_headless(
            result,
            database,
            storage,
            DeduplicationService(database),
            TranslationService(database, semantic=pipeline.semantic),
        )
        if not save_problems:
            database.update_staging_session(session_id, status="CONFIRMED")
            return _json(
                {
                    "status": "ok",
                    "auto_commit": True,
                    "session_id": session_id,
                    "saved": saved,
                    "total": summary["total"],
                    "warnings": [*result.warnings, *extra_warnings],
                    "message": (f"{len(saved)} question(s) automatically saved to the database."),
                }
            )
        problems = save_problems

    review_url = f"{settings.UI_PUBLIC_URL.rstrip('/')}/?session_id={session_id}"
    status = STATUS_NEEDS_ATTENTION if problems else result.status
    if problems:
        message = "there are issues; review the session in the UI: " + review_url
    else:
        message = (
            f"{summary['valid']} valid question(s) in the staging session; "
            "review and confirm in the UI: " + review_url
        )
    return _json(
        {
            "status": status,
            "session_id": session_id,
            "review_url": review_url,
            "total": summary["total"],
            "valid": summary["valid"],
            "needs_attention": summary["needs_attention"],
            "problems": problems,
            "warnings": [*result.warnings, *extra_warnings],
            "message": message,
        }
    )


@mcp.tool()
def refine_staged_question(
    session_id: str,
    q_index: int,
    user_hint: str,
    manual_overrides: dict[str, Any] | None = None,
) -> str:
    """Reanalyzes a staging question with corrective context.

    Applies the manual overrides (statement, type, choices and curricular
    classification) and preserves them explicitly. It does not solve the
    exercise.
    """
    try:
        _settings, database, _storage, pipeline, _exams = _resources()
        session = database.get_staging_session(session_id)
        if session is None:
            return _json({"status": "error", "message": f"session not found: {session_id}"})
        data = session["data"]
        questions = data.get("questions", [])
        if q_index < 0 or q_index >= len(questions):
            return _json({"status": "error", "message": f"index out of range: {q_index}"})

        overrides = manual_overrides or {}
        raw_overrides = {
            key: value for key, value in overrides.items() if key in _ALLOWED_OVERRIDES
        }
        curricular = {
            key: value for key, value in overrides.items() if key in _CURRICULAR_OVERRIDES
        }

        intermediate = _intermediate_from_session(data)
        current_payload = questions[q_index].get("question")
        if current_payload is None:
            return _json(
                {
                    "status": "error",
                    "message": f"question {q_index} has no statement to reanalyze",
                }
            )
        current = EnrichedQuestion.model_validate(current_payload)
        refined = pipeline.refine_question(intermediate, current, hints=user_hint or None)
        if refined.question is not None:
            refined.question = _apply_overrides(refined.question, raw_overrides)
            refined = pipeline.validate_question(
                refined.question,
                max_pages=len(intermediate.pages) or None,
                taxonomy_names=refined.taxonomy_names,
            )
            payload = refined.question.model_dump()
            payload.update(curricular)
            payload.setdefault("created_at", utc_now_iso())
        else:
            payload = None
        questions[q_index] = {
            "status": refined.status,
            "reasons": refined.reasons,
            "question": payload,
            "taxonomy_names": [names.model_dump() for names in refined.taxonomy_names],
        }
        data["questions"] = questions
        database.update_staging_session(session_id, data=data)
        return _json(
            {
                "status": refined.status,
                "q_index": q_index,
                "reasons": refined.reasons,
                "question": payload,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def confirm_staged_upload(session_id: str) -> str:
    """Confirms a staging session and persists the approved questions.

    It is idempotent: a session that is already confirmed is reported without
    reinserting.
    """
    try:
        _settings, database, storage, _pipeline, _exams = _resources()
        dedup = DeduplicationService(database)
        session = database.get_staging_session(session_id)
        if session is None:
            return _json({"status": "error", "message": f"session not found: {session_id}"})
        if session["status"] == "CONFIRMED":
            return _json({"status": "ok", "message": f"session {session_id} was already confirmed"})

        saved: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        translations = TranslationService(database, semantic=_pipeline.semantic)
        for item in session["data"].get("questions", []):
            if item.get("status") != STATUS_VALID:
                skipped.append(f"{item.get('question', {}).get('id', '?')} (not VALID)")
                continue
            payload = item.get("question")
            if not payload:
                continue
            question = _enrich(payload, storage)
            if question is None:
                skipped.append(f"{payload.get('id', '?')} (no curricular classification)")
                continue
            try:
                database.add_prerequisites(list(question.prerequisites))
                database.upsert_course(question.course)
                database.upsert_topic(question.course, question.topic, question.subtopic)
                _persist_taxonomy_names(
                    database,
                    _taxonomy_names_from_payload(item.get("taxonomy_names")),
                    question,
                )
                database.upsert_question(question, embedding=dedup.embed_question(question))
                warnings.extend(translations.store(question).problems)
                saved.append(question.id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{payload.get('id', '?')}: {exc}")

        if saved:
            database.update_staging_session(session_id, status="CONFIRMED")
        return _json(
            {
                "status": "ok" if saved else "error",
                "saved": saved,
                "skipped": skipped,
                "errors": errors,
                "warnings": warnings,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# MCP tools — queries and RAG
# ---------------------------------------------------------------------------


@mcp.tool()
def search_questions(
    course: str = "",
    topic: str = "",
    difficulty: int = 0,
    limit: int = 20,
    locale: str = "",
) -> str:
    """Retrieves real questions by course, topic and difficulty (without solutions).

    By default each question includes ``statement_by_locale`` with all the
    localized copies; with ``locale`` (``es``/``en``) it also includes the
    ``statement`` already resolved for that language.
    """
    try:
        settings, database, _storage, _pipeline, _exams = _resources()
        limit = max(1, min(limit, _MAX_SEARCH_LIMIT))
        active = locale if is_supported_locale(locale) else None
        questions = database.search_questions(
            course=course or None,
            topic=topic or None,
            difficulty=difficulty if difficulty >= 1 else None,
            limit=limit,
        )
        return _json(
            {
                "count": len(questions),
                "locale": active,
                "questions": [
                    _public_question(q, settings, database, locale=active) for q in questions
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def list_taxonomy(locale: str = "") -> str:
    """Queries the catalog of courses, topics, subtopics and prerequisites.

    By default it includes a ``names`` map with all the locales; with ``locale``
    (``es``/``en``) it returns a flat shape with a single ``name`` per entry.
    """
    try:
        _settings, database, _storage, _pipeline, _exams = _resources()
        return _json(database.list_taxonomy(locale=locale or None))
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# MCP tools — generation and compilation
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_practice_exam(
    course: str = "",
    topics: list[str] | None = None,
    difficulty: int = 0,
    num_questions: int = 10,
    locale: str = "",
) -> str:
    """Generates and compiles a practice exam as a PDF with Typst.

    With ``locale`` (``es``/``en``) the translated copy of the statement is used
    when it exists, along with the localized course name.
    """
    try:
        _settings, _database, _storage, _pipeline, exams = _resources()
        result = exams.generate_exam(
            course=course or None,
            topics=topics or None,
            difficulty=difficulty if difficulty >= 1 else None,
            num_questions=num_questions,
            locale=locale or None,
        )
        return _json(result.summary())
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def get_exam_source(exam_id: str, format: str = "typst") -> str:
    """Generates and returns the URL of a ZIP with the exam source."""
    try:
        settings, _database, _storage, _pipeline, exams = _resources()
        zip_path = exams.build_source_zip(exam_id, format=format)
        url = f"{settings.DOWNLOAD_PUBLIC_URL.rstrip('/')}/download/{zip_path.name}"
        return _json({"exam_id": exam_id, "format": format, "zip_url": url, "path": str(zip_path)})
    except (ExamGenerationError, ValueError) as exc:
        return _json({"status": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return _json({"status": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Internal staging helpers
# ---------------------------------------------------------------------------


def _intermediate_from_session(data: dict[str, Any]) -> TranscriptionIntermediate:
    intermediate = data.get("intermediate")
    if intermediate is not None:
        return TranscriptionIntermediate.model_validate(intermediate)
    questions = data.get("questions", [])
    first = questions[0].get("question") if questions else None
    return TranscriptionIntermediate(
        source_reference=data.get("source_reference", "staging"),
        raw_markdown=first.get("statement_raw", "") if first else "",
        pages=[],
        figures=[],
        warnings=[],
    )


def _taxonomy_names_from_payload(payload: Any) -> list[TaxonomyNames]:
    """Rebuilds taxonomy names from a staging payload (tolerant).

    The names are optional presentation data: a malformed entry is dropped
    instead of invalidating the confirmation.
    """
    if not isinstance(payload, list):
        return []
    names: list[TaxonomyNames] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            names.append(TaxonomyNames.model_validate(item))
        except Exception:  # noqa: BLE001 - optional presentation, never blocks
            continue
    return names


def _persist_taxonomy_names(
    database: Database,
    names: list[TaxonomyNames],
    question: EnrichedQuestion,
) -> None:
    """Persists the LLM names that still match the final classification.

    It is written with ``source='llm'`` so as not to overwrite names that were
    seeded or edited by hand. Names of entries whose classification changed
    during review are dropped.
    """
    applicable = filter_taxonomy_names(
        names,
        course=question.course,
        topic=question.topic,
        subtopic=question.subtopic,
        prerequisites=question.prerequisites,
    )
    if applicable:
        database.upsert_taxonomy_names(applicable)


def _save_headless(
    result: StagingResult,
    database: Database,
    storage: ObjectStorage,
    dedup: DeduplicationService,
    translations: TranslationService,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Automatically persists the VALID questions of a headless ingestion.

    Uses :class:`DeduplicationService` to detect semantic duplicates (by
    embeddings and, optionally, an LLM judge) and captures any exception as
    ``ERROR``. Returns the saved ids, the list of issues and a list of warnings
    (e.g. translations that could not be generated or validated).
    """
    saved: list[str] = []
    problems: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item in result.questions:
        question = item.question
        if item.status != STATUS_VALID or question is None:
            problems.append(
                {
                    "id": question.id if question is not None else None,
                    "status": item.status,
                    "reasons": item.reasons,
                }
            )
            continue
        try:
            enriched = _enrich(question.model_dump(), storage)
            if enriched is None:
                problems.append(
                    {
                        "id": question.id,
                        "status": "ERROR",
                        "reasons": ["no curricular classification"],
                    }
                )
                continue
            database.add_prerequisites(list(enriched.prerequisites))
            database.upsert_course(enriched.course)
            database.upsert_topic(enriched.course, enriched.topic, enriched.subtopic)
            _persist_taxonomy_names(database, item.taxonomy_names, enriched)
            outcome = dedup.ingest(enriched)
        except Exception as exc:  # noqa: BLE001
            problems.append({"id": question.id, "status": "ERROR", "reasons": [str(exc)]})
            continue
        if outcome.status == "DUPLICATE":
            problems.append(
                {
                    "id": question.id,
                    "status": "DUPLICATE",
                    "reasons": [outcome.reason or "duplicate question"],
                }
            )
        else:
            saved.append(question.id)
            warnings.extend(translations.store(enriched).problems)
    return saved, problems, warnings


def _enrich(payload: dict[str, Any], storage: ObjectStorage) -> EnrichedQuestion | None:
    """Builds an ``EnrichedQuestion`` from the staging payload.

    Requires curricular classification (course/topic/difficulty) and verifies
    that the figures are in the persistent bucket before persisting.
    """
    payload = dict(payload)
    for key in ("course", "topic", "difficulty"):
        if key not in payload:
            return None
    payload.setdefault("created_at", utc_now_iso())
    payload.setdefault("source_metadata", {})
    question = EnrichedQuestion.model_validate(payload)
    for figure in question.figures:
        if not figure.asset_path:
            raise ValueError(f"figure {figure.id} has no persistent identifier")
        bucket, key = split_asset_reference(figure.asset_path)
        if bucket:
            if not storage.exists(key, bucket=bucket):
                raise ValueError(f"figure {figure.id} is not in the bucket")
        elif not storage.exists(figure.asset_path):
            raise ValueError(f"figure {figure.id} is not in the bucket")
    return question


# ---------------------------------------------------------------------------
# HTTP routes and TTL cleaner
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> Response:
    """Health endpoint for Docker Compose healthchecks."""
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/download/{filename:path}", methods=["GET"])
async def download_file(request: Request) -> Response:
    """Serves a temporary file from the downloads directory."""
    settings = get_settings()
    filename = request.path_params.get("filename", "")
    try:
        path = _resolve_download_path(filename, settings)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    media_type = "application/pdf" if path.suffix == ".pdf" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@mcp.custom_route("/assets/{asset_id:path}", methods=["GET"])
async def asset_proxy(request: Request) -> Response:
    """Returns a figure from SeaweedFS by its controlled identifier.

    It is not an arbitrary S3 proxy: the identifier must be a valid key or
    s3:// URI, and it is validated against the configured storage.
    """
    settings = get_settings()
    asset_id = request.path_params.get("asset_id", "")
    storage = ObjectStorage(settings)
    try:
        bucket, key = storage.resolve(asset_id)
        data = storage.get_bytes(key, bucket=bucket)
    except ObjectStorageError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    content_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(Path(key).suffix.lower(), "application/octet-stream")
    return Response(content=data, media_type=content_type)


def _clean_downloads(settings: Settings) -> int:
    """Removes temporary files from ``DOWNLOADS_DIR`` older than ``TTL`` seconds."""
    downloads = Path(settings.DOWNLOADS_DIR).expanduser()
    if not downloads.is_dir():
        return 0
    cutoff = time.time() - settings.DOWNLOADS_TTL_SECONDS
    removed = 0
    for path in downloads.rglob("*"):
        if path.is_file():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:  # noqa: BLE001
                logger.warning("could not delete %s: %s", path, exc)
    return removed


def start_download_cleaner(
    settings: Settings | None = None, *, interval_seconds: int = 300
) -> threading.Thread:
    """Starts a background thread that cleans expired downloads.

    It only touches files in ``DOWNLOADS_DIR``: never the database, the original
    uploads nor the persistent objects in the bucket.
    """
    resolved = settings or get_settings()

    def _loop() -> None:
        while True:
            try:
                removed = _clean_downloads(resolved)
                if removed:
                    logger.info("download cleaner removed %d file(s)", removed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("error in the download cleaner: %s", exc)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="taygeta-download-cleaner", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Starts the MCP server with the HTTP transport and the TTL cleaner."""
    settings = get_settings()
    start_download_cleaner(settings)
    logger.info("Taygeta MCP server listening on %s:%s", settings.MCP_HOST, settings.MCP_PORT)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
