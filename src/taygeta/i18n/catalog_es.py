"""Spanish (es) locale catalog for the Taygeta interface.

Keys are dotted namespaces (``area.subject``) and must stay in sync with the
other catalogs; :mod:`tests.test_i18n` enforces key parity across locales.
Placeholders in ``{braces}`` are interpolated by :func:`taygeta.i18n.t`.
"""

from __future__ import annotations

CATALOG: dict[str, str] = {
    # -- Language switcher --------------------------------------------------
    "ui.language.tooltip": "Cambiar el idioma de la interfaz",
    # -- Main page ----------------------------------------------------------
    "ui.page.title": "Taygeta — Revisión HITL",
    "ui.app.heading": "Taygeta — Ingesta y revisión Human-in-the-Loop",
    "ui.app.upload_label": "PDF, PNG o JPG",
    "ui.app.upload_failed": "No se pudo guardar el archivo: {error}",
    "ui.app.upload_saved": "Guardado en uploads: {name}",
    "ui.app.section.upload": "1. Subir documento",
    "ui.app.section.process": "2. Procesar",
    "ui.app.section.review": "3. Revisión y confirmación",
    "ui.app.doc_select": "Documento a procesar",
    "ui.app.hint": "Contexto para el análisis (opcional)",
    "ui.app.process_button": "Procesar documento",
    "ui.app.doc_select_warning": "Selecciona un documento de la lista",
    "ui.app.processing": (
        "Procesando {name}: rasterización, OCR local, estructuración y validación…"
    ),
    "ui.app.ingest_error": "Error de ingesta: {error}",
    "ui.app.ingest_cancelled": "La ingesta fue cancelada.",
    "ui.app.ingest_done": "Ingesta completada: {status}",
    "ui.status.title": "Estado",
    "ui.status.total": "Total",
    "ui.status.db_line": (
        "- **BD**: `{db_path}` · integridad: {integrity} · FTS5: {fts5} · preguntas: {questions}"
    ),
    "ui.status.db_unavailable": "Base de datos no disponible: {error}",
    "ui.status.llm_unconfigured": (
        "LLM remoto sin configurar: la validación semántica quedará en NEEDS_ATTENTION."
    ),
    "ui.status.taxonomy": "Taxonomía",
    "ui.status.courses": "Cursos: {courses}",
    "ui.status.prerequisites": "Prerrequisitos: {prerequisites}",
    "ui.review.session": "Sesión de staging: `{session_id}`",
    "ui.review.warnings": "Advertencias de OCR",
    "ui.review.note": (
        "La taxonomía, la dificultad y los prerrequisitos se infirieron automáticamente "
        "durante la estructuración; revísalos y edítalos antes de confirmar."
    ),
    "ui.review.confirm_button": "Confirmar y persistir preguntas aprobadas",
    "ui.review.nothing_approved": "No hay preguntas aprobadas para guardar.",
    "ui.review.invalid_slugs": (
        "Curso y tema deben ser slugs válidos (minúsculas, dígitos y guion bajo)."
    ),
    "ui.review.taxonomy_error": "No se pudo actualizar la taxonomía: {error}",
    "ui.review.saved": "Persistidas {count} pregunta(s) en taygeta.db.",
    "ui.dark_mode.tooltip": "Cambiar entre modo claro y oscuro",
    # -- Question panel -----------------------------------------------------
    "ui.panel.header": "Pregunta {number} — {status}",
    "ui.panel.valid": "Validación superada (VALID). Revísala y apruébala para guardarla.",
    "ui.panel.attention": "NEEDS_ATTENTION: {reasons}",
    "ui.panel.rejection": "Motivo de rechazo semántico: {reason}",
    "ui.panel.no_statement": "No hay enunciado estructurado para editar en esta pregunta.",
    "ui.panel.tab.original": "Imagen original",
    "ui.panel.tab.figures": "Figuras nuevas",
    "ui.panel.approve": "Aprobar y guardar esta pregunta",
    "ui.panel.statement": "Enunciado (Markdown/TeX)",
    "ui.panel.qtype": "Tipo de pregunta",
    "ui.panel.notation_notes": "Notas de notación: {notes}",
    "ui.panel.metadata": "Metadatos",
    "ui.panel.choice.id": "Alternativa {number} — id",
    "ui.panel.choice.content": "Alternativa {number} — contenido",
    "ui.panel.course": "Curso",
    "ui.panel.course.tooltip": "Curso del catálogo o escribe uno nuevo manualmente.",
    "ui.panel.topic": "Tema",
    "ui.panel.topic.tooltip": "Temas del curso seleccionado o escribe uno nuevo manualmente.",
    "ui.panel.subtopic": "Subtema (slug snake_case, opcional)",
    "ui.panel.difficulty": "Dificultad: {value}",
    "ui.panel.prerequisites": "Prerrequisitos",
    "ui.panel.advanced": "Opciones Avanzadas",
    "ui.panel.hint": "Contexto correctivo (p. ej. «La condición ilegible es g(x) = ...»)",
    "ui.panel.refine": "Reanalizar enunciado",
    "ui.panel.new_prereqs": "Nuevos prerrequisitos (separados por coma)",
    "ui.panel.dangling": (
        "El enunciado referencia placeholders sin figura declarada: {placeholders}. "
        "Elimina el placeholder o restaura la figura para poder aprobar."
    ),
    "ui.panel.figures_title": "Figuras de la pregunta",
    "ui.panel.figure.load_failed": "{figure_id}: no se pudo cargar el recorte (página {page})",
    "ui.panel.figure.caption": "{figure_id} — recorte (página {page})",
    "ui.panel.figure.open": "Abrir {figure_id} en el navegador",
    "ui.panel.figure.delete": "Eliminar figura",
    "ui.panel.figure.recrop": "Recortar manualmente — {figure_id}",
    "ui.panel.figure.recrop_none": "No hay documento de origen para recortar esta figura.",
    "ui.panel.figure.recrop_caption": "Arrastra sobre la página {page} para ajustar el recorte",
    "ui.panel.figure.apply_crop": "Aplicar recorte",
    "ui.panel.figure.select_region_warning": (
        "Arrastra sobre la página para seleccionar la región"
    ),
    "ui.panel.figure.crop_failed": "No se pudo aplicar el recorte de {figure_id}: {error}",
    "ui.panel.figure.cropped": "{figure_id} recortada y guardada.",
    "ui.panel.figure.add_expansion": "➕ Añadir nueva figura",
    "ui.panel.figure.no_source_add": "No hay documento de origen para añadir figuras.",
    "ui.panel.figure.page": "Página {number}",
    "ui.panel.figure.page_label": "Página original del documento",
    "ui.panel.figure.page_failed": "No se pudo cargar la página original para recortar.",
    "ui.panel.figure.add_button": "Añadir figura",
    "ui.panel.figure.add_failed": "No se pudo añadir la figura: {error}",
    "ui.panel.figure.added": ("{figure_id} añadida; su placeholder quedó al final del enunciado."),
    "ui.panel.figure.delete_failed": "No se pudo eliminar la figura: {error}",
    "ui.panel.original.none": "No hay documento de origen para mostrar.",
    "ui.panel.original.page": "Página original {number}",
    "ui.panel.original.page_failed": "No se pudo rasterizar la página {number}.",
    "ui.panel.refine_failed": "No se pudo reanalizar: {error}",
    # -- Figure cropper -----------------------------------------------------
    "ui.cropper.select_region": "Arrastra para seleccionar la región",
    "ui.cropper.select_figure": "Arrastra y suelta para seleccionar la figura",
    "ui.cropper.zoom_reset": "Ajustar a la vista",
    # -- Persistence (surfaced as notifications) ----------------------------
    "ui.persist.figure_missing_asset": "{figure_id} (sin asset_path)",
    "ui.persist.figure_not_in_bucket": (
        "{question_id}: figuras no presentes en el bucket: {figures}"
    ),
    "ui.persist.session_update_failed": "no se pudo marcar la sesión como confirmada: {error}",
    # -- Generated exam documents -------------------------------------------
    "exam.title": "Examen de práctica",
    "exam.title_with_course": "Examen de práctica — {course}",
    "exam.question_heading": "Pregunta {index}",
    "exam.latex.babel_language": "spanish",
    "exam.readme.latex": (
        "Fuente del examen en LaTeX.\n\n"
        "Compilar con: pdflatex main.tex\n\n"
        "Las figuras utilizadas están en la carpeta assets/.\n"
    ),
    "exam.readme.typst": (
        "Fuente del examen en Typst.\n\n"
        "Compilar con: typst compile main.typ output.pdf\n\n"
        "Las figuras utilizadas están en la carpeta assets/.\n"
    ),
}
