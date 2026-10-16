"""Aplicación NiceGUI de revisión Human-in-the-Loop de Taygeta.

Modelo de estado por eventos (sin reruns): la página se construye una vez por
sesión y los manejadores mutan objetos Python que refrescan sus contenedores.
El flujo es:

1. subida de PDF/PNG/JPG al directorio ``UPLOADS_DIR``;
2. procesamiento con el pipeline de doble pase en background
   (``run.io_bound``, sin bloquear la interfaz);
3. revisión por pregunta: editor Markdown/TeX con previsualización, taxonomía
   con comboboxes de curso/tema (catálogo o escritura manual), figuras a la
   derecha con pestañas «Imagen original» / «Figuras nuevas» y cropper de
   arrastre, y aprobación explícita antes de persistir en SQLite
   (``taygeta.db``). Un botón flotante (arriba a la derecha) alterna el modo
   claro/oscuro y conserva la preferencia entre recargas.

Se ejecuta con ``python -m taygeta.ui.nice`` o con el entry point
``taygeta-ui``.

Todo el texto visible se resuelve con :func:`taygeta.i18n.t` (catálogos
``es``/``en``); las cadenas se mantienen en los catálogos, no en este módulo.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path
from typing import Any

from nicegui import app, run, ui

from taygeta.core.schema import (
    EnrichedQuestion,
    FigureReference,
    sanitize_math_delimiters,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.taxonomy import TaxonomyNames, filter_taxonomy_names, format_course_name
from taygeta.i18n import get_locale, t
from taygeta.ingest.pipeline import IngestPipeline
from taygeta.server.db import Database, DatabaseError
from taygeta.server.object_storage import ObjectStorage
from taygeta.server.translation_service import TranslationService
from taygeta.ui.nice import caption
from taygeta.ui.nice.localization import (
    install_ui_language,
    language_switcher,
    quasar_locale,
)
from taygeta.ui.nice.math_render import install_math_rendering
from taygeta.ui.nice.question_panel import PanelDraft, QuestionPanel
from taygeta.ui.nice.state import QuestionState, SessionState, drop, register
from taygeta.ui.nice.taxonomy import install_taxonomy_names, invalidate_taxonomy_names
from taygeta.ui.shared import (
    build_enriched_question,
    delete_asset,
    list_uploads,
    persist_approved_questions,
    public_s3_url,
    save_upload,
)

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@functools.lru_cache(maxsize=1)
def _resources() -> tuple[Settings, Database, ObjectStorage, IngestPipeline]:
    """Recursos compartidos por todas las sesiones (análogo a ``cache_resource``)."""
    settings = get_settings()
    database = Database(settings=settings)
    storage = ObjectStorage(settings)
    pipeline = IngestPipeline(settings, storage=storage, database=database)
    # Los nombres de taxonomía se resuelven desde la base de datos.
    install_taxonomy_names(database)
    return settings, database, storage, pipeline


@app.get("/health")
def _health() -> dict[str, str]:
    """Healthcheck simple para el orquestador (docker-compose)."""
    return {"status": "ok"}


def _refresh_docs(doc_select: ui.select, settings: Settings) -> None:
    """Actualiza el selector de documentos con el directorio de uploads."""
    names = [path.name for path in list_uploads(settings.UPLOADS_DIR)]
    current = doc_select.value
    doc_select.options = names
    if current not in names:
        doc_select.value = names[0] if names else None
    doc_select.update()


def _stat(label: str, value: str) -> None:
    """Mini-tarjeta de métrica para el resumen de la ingesta."""
    with ui.card().classes("w-28 items-center"):
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-gray-500")


async def _on_upload(
    e: Any, session: SessionState, doc_select: ui.select, settings: Settings
) -> None:
    """Guarda un archivo subido en ``UPLOADS_DIR`` y refresca el selector."""
    try:
        data = await e.file.read()
        path = save_upload(e.file.name, data, settings.UPLOADS_DIR)
    except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
        ui.notify(t("ui.app.upload_failed", error=exc), type="negative")
        return
    _refresh_docs(doc_select, settings)
    ui.notify(t("ui.app.upload_saved", name=path.name), type="positive")


def _build_review(
    session: SessionState,
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> list[QuestionPanel]:
    """Dibuja el resumen y los paneles de revisión; devuelve los paneles creados."""
    result = session.staging
    if result is None:
        return []
    panels: list[QuestionPanel] = []
    summary = result.summary()
    with ui.row().classes("w-full gap-4"):
        _stat(t("ui.status.title"), summary["status"])
        _stat(t("ui.status.total"), str(summary["total"]))
        _stat("VALID", str(summary["valid"]))
        _stat("NEEDS_ATTENTION", str(summary["needs_attention"]))
    if summary["session_id"]:
        caption(t("ui.review.session", session_id=summary["session_id"]))
    if result.warnings:
        with ui.expansion(t("ui.review.warnings")):
            ui.markdown("\n".join(f"- {warning}" for warning in result.warnings))
    caption(t("ui.review.note"))

    source_path = session.processed_path
    for index, staged in enumerate(result.questions):
        panels.append(
            QuestionPanel(
                ui.column().classes("w-full"),
                index=index,
                staged=staged,
                session=session,
                source_path=source_path,
                storage=storage,
                settings=settings,
                database=database,
                pipeline=pipeline,
            )
        )
    ui.separator()
    ui.button(
        t("ui.review.confirm_button"),
        color="primary",
        on_click=lambda: _confirm(session, panels, database, storage, pipeline),
    )
    return panels


async def _on_process(
    session: SessionState,
    doc_select: ui.select,
    hint_input: ui.input,
    status_row: ui.row,
    status_label: ui.label,
    review: ui.column,
    settings: Settings,
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> None:
    """Procesa el documento seleccionado y (re)construye la revisión."""
    if session.processing:
        return
    if not doc_select.value:
        ui.notify(t("ui.app.doc_select_warning"), type="warning")
        return
    path = Path(settings.UPLOADS_DIR) / str(doc_select.value)
    if str(path) != session.processed_path:
        session.staging = None
        session.questions = {}
        session.orphans = {}
    session.processing = True
    status_row.set_visibility(True)
    status_label.set_text(t("ui.app.processing", name=path.name))
    try:
        result = await run.io_bound(pipeline.ingest, path, hints=hint_input.value or None)
    except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
        session.processing = False
        status_row.set_visibility(False)
        ui.notify(t("ui.app.ingest_error", error=exc), type="negative")
        return
    session.processing = False
    status_row.set_visibility(False)
    if result is None:
        ui.notify(t("ui.app.ingest_cancelled"), type="warning")
        return
    session.processed_path = str(path)
    session.staging = result
    session.questions = {index: QuestionState() for index in range(len(result.questions))}
    review.clear()
    with review:
        _build_review(session, settings, database, storage, pipeline)
    ui.notify(t("ui.app.ingest_done", status=result.status), type="positive")


def _confirm(
    session: SessionState,
    panels: list[QuestionPanel],
    database: Database,
    storage: ObjectStorage,
    pipeline: IngestPipeline,
) -> None:
    """Recoge los paneles aprobados y persiste las preguntas en SQLite.

    Revalida los slugs de curso/tema y purga los assets huérfanos de las
    preguntas efectivamente confirmadas (nunca borra figuras de preguntas no
    confirmadas).
    """
    result = session.staging
    if result is None:
        return
    approved: list[tuple[EnrichedQuestion, list[str], list[TaxonomyNames]]] = []
    _drafts: list[PanelDraft] = []
    for panel in panels:
        draft = panel.collect()
        if draft is None:
            continue
        _drafts.append(draft)
        if not draft.approved or draft.colgantes:
            continue
        approved_question = build_enriched_question(
            draft.base,
            course=draft.course,
            topic=draft.topic,
            subtopic=draft.subtopic,
            difficulty=draft.difficulty,
            prerequisites=[*draft.prerequisites, *draft.new_prereqs],
            source_metadata={"source_reference": result.source_reference},
            figures=[FigureReference.model_validate(figure) for figure in draft.effective_figures],
        )
        # Reflejar la edición manual (sanitiza delimitadores TeX no estándar).
        approved_question = approved_question.model_copy(
            update={
                "statement_raw": sanitize_math_delimiters(draft.statement),
                "question_type": draft.qtype,
                "choices": draft.choices,
            }
        )
        names = filter_taxonomy_names(
            panel.staged.taxonomy_names,
            course=approved_question.course,
            topic=approved_question.topic,
            subtopic=approved_question.subtopic,
            prerequisites=approved_question.prerequisites,
        )
        approved.append((approved_question, draft.new_prereqs, names))
    if not approved:
        ui.notify(t("ui.review.nothing_approved"), type="warning")
        return
    invalid = [
        q for q, _, _ in approved if not _SLUG_RE.match(q.course) or not _SLUG_RE.match(q.topic)
    ]
    if invalid:
        ui.notify(t("ui.review.invalid_slugs"), type="negative")
        return
    new_tags = sorted({slug for _, slugs, _ in approved for slug in slugs})
    try:
        for question, _, names in approved:
            database.upsert_course(question.course)
            database.upsert_topic(question.course, question.topic, question.subtopic)
            database.upsert_taxonomy_names(names)
        if new_tags:
            database.add_prerequisites(new_tags)
    except DatabaseError as exc:
        ui.notify(t("ui.review.taxonomy_error", error=exc), type="negative")
        return
    invalidate_taxonomy_names()
    confirm_result = persist_approved_questions(
        [question for question, _, _ in approved],
        database,
        storage,
        session_id=result.session_id,
        translations=TranslationService(database, semantic=pipeline.semantic),
    )
    _purge_orphans(session, set(confirm_result.saved), storage)
    if confirm_result.saved:
        ui.notify(t("ui.review.saved", count=len(confirm_result.saved)), type="positive")
    for error in confirm_result.errors:
        ui.notify(error, type="negative")
    for warning in confirm_result.warnings:
        ui.notify(warning, type="warning")


def _purge_orphans(session: SessionState, confirmed_ids: set[str], storage: ObjectStorage) -> None:
    """Borra los assets huérfanos de las preguntas efectivamente confirmadas."""
    for question_id in confirmed_ids:
        for asset_path in session.orphans.get(str(question_id), set()):
            delete_asset(storage, asset_path)
    session.orphans = {}


def _dark_mode_toggle() -> None:
    """Botón flotante (esquina superior derecha) que alterna modo claro/oscuro.

    El estado se guarda en ``app.storage.user`` (cookie de sesión) para
    conservarlo entre recargas. Requiere ``storage_secret`` en ``ui.run``.
    """
    stored = bool(app.storage.user.get("dark_mode", False))
    dark = ui.dark_mode(value=stored)

    def toggle() -> None:
        dark.toggle()
        app.storage.user["dark_mode"] = bool(dark.value)
        button.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")

    button = (
        ui.button(icon="light_mode" if stored else "dark_mode", on_click=toggle)
        .classes("fixed top-4 right-4 z-50")
        .props("round flat color=primary")
    )
    button.mark("toggle-modo-oscuro")
    button.tooltip(t("ui.dark_mode.tooltip"))


@ui.page("/")
def _index_page() -> None:
    settings, database, storage, pipeline = _resources()
    session = register(ui.context.client.id)
    ui.page_title(t("ui.page.title"))
    # Carga KaTeX (recuadro de formato del enunciado) en el <head> de la página.
    install_math_rendering()
    _dark_mode_toggle()
    language_switcher()
    install_ui_language(get_locale())

    # Contenedor amplio: en pantallas anchas se aprovecha el espacio disponible
    # (hasta 1800 px) con poco margen lateral, en lugar del antiguo max-w-7xl.
    with ui.column().classes("w-full max-w-[1800px] mx-auto gap-2 px-2 py-4"):
        ui.label(t("ui.app.heading")).classes("text-2xl font-bold")

        # --- Estado y taxonomía ----------------------------------------
        with ui.expansion(t("ui.status.title")):
            try:
                status = database.status()
                ui.markdown(
                    t(
                        "ui.status.db_line",
                        db_path=status["db_path"],
                        integrity=status["integrity"],
                        fts5=status["fts5_enabled"],
                        questions=status["counts"]["questions"],
                    )
                )
            except DatabaseError as exc:
                ui.label(t("ui.status.db_unavailable", error=exc)).classes("text-red-600")
            endpoint = public_s3_url(settings.S3_ENDPOINT_URL, settings)
            caption(f"S3: {settings.S3_BUCKET} @ {endpoint}")
            if not settings.llm_configured:
                ui.label(t("ui.status.llm_unconfigured")).classes("text-amber-600")
            with ui.expansion(t("ui.status.taxonomy")):
                ui.label(
                    t(
                        "ui.status.courses",
                        courses=", ".join(
                            format_course_name(c["slug"]) for c in database.list_courses()
                        ),
                    )
                )
                ui.label(
                    t(
                        "ui.status.prerequisites",
                        prerequisites=", ".join(p["slug"] for p in database.list_prerequisites()),
                    )
                )

        # --- 1. Subida ---------------------------------------------------
        ui.label(t("ui.app.section.upload")).classes("text-lg font-semibold")
        ui.upload(
            label=t("ui.app.upload_label"),
            auto_upload=True,
            on_upload=lambda e: _on_upload(e, session, doc_select, settings),
        )

        # --- 2. Procesar -------------------------------------------------
        ui.label(t("ui.app.section.process")).classes("text-lg font-semibold")
        with ui.row().classes("w-full gap-4 items-center"):
            doc_select = ui.select([], label=t("ui.app.doc_select")).classes(
                "w-80 flex-1 max-w-full"
            )
            hint_input = ui.input(t("ui.app.hint")).classes("w-full")
            process_btn = ui.button(t("ui.app.process_button"), color="primary")
        status_row = ui.row().classes("items-center gap-2")
        with status_row:
            ui.spinner(size="lg")
            status_label = ui.label("").classes("text-gray-600")
        status_row.set_visibility(False)
        process_btn.on_click(
            lambda: _on_process(
                session,
                doc_select,
                hint_input,
                status_row,
                status_label,
                review,
                settings,
                database,
                storage,
                pipeline,
            )
        )
        _refresh_docs(doc_select, settings)

        # --- 3. Revisión y confirmación ----------------------------------
        ui.label(t("ui.app.section.review")).classes("text-lg font-semibold")
        review = ui.column().classes("w-full gap-4")
        if session.staging is not None:
            with review:
                _build_review(session, settings, database, storage, pipeline)


def main() -> None:
    """Arranca el servidor NiceGUI de la revisión HITL."""
    settings = get_settings()
    host = os.getenv("TAYGETA_UI_HOST", "0.0.0.0")
    port = int(os.getenv("TAYGETA_UI_PORT", "8501"))
    ui.run(
        host=host,
        port=port,
        title=t("ui.page.title", locale=settings.UI_LANGUAGE),
        show=False,
        reload=False,
        language=quasar_locale(settings.UI_LANGUAGE),
        # Necesario para app.storage.user (persistencia del modo oscuro).
        storage_secret=os.getenv("TAYGETA_UI_STORAGE_SECRET", "taygeta-hitl-local"),
    )


app.on_disconnect(lambda client: drop(client.id))
