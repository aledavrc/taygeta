"""English (en) locale catalog for the Taygeta interface.

Keys are dotted namespaces (``area.subject``) and must stay in sync with the
other catalogs; :mod:`tests.test_i18n` enforces key parity across locales.
Placeholders in ``{braces}`` are interpolated by :func:`taygeta.i18n.t`.
"""

from __future__ import annotations

CATALOG: dict[str, str] = {
    # -- Language switcher --------------------------------------------------
    "ui.language.tooltip": "Change the interface language",
    # -- Main page ----------------------------------------------------------
    "ui.page.title": "Taygeta — HITL review",
    "ui.app.heading": "Taygeta — Human-in-the-Loop ingestion and review",
    "ui.app.upload_label": "PDF, PNG or JPG",
    "ui.app.upload_failed": "Could not save the file: {error}",
    "ui.app.upload_saved": "Saved to uploads: {name}",
    "ui.app.section.upload": "1. Upload document",
    "ui.app.section.process": "2. Process",
    "ui.app.section.review": "3. Review and confirmation",
    "ui.app.doc_select": "Document to process",
    "ui.app.hint": "Analysis context (optional)",
    "ui.app.process_button": "Process document",
    "ui.app.doc_select_warning": "Select a document from the list",
    "ui.app.processing": (
        "Processing {name}: rasterization, local OCR, structuring and validation…"
    ),
    "ui.app.ingest_error": "Ingestion error: {error}",
    "ui.app.ingest_cancelled": "Ingestion was cancelled.",
    "ui.app.ingest_done": "Ingestion completed: {status}",
    "ui.status.title": "Status",
    "ui.status.total": "Total",
    "ui.status.db_line": (
        "- **DB**: `{db_path}` · integrity: {integrity} · FTS5: {fts5} · questions: {questions}"
    ),
    "ui.status.db_unavailable": "Database unavailable: {error}",
    "ui.status.llm_unconfigured": (
        "Remote LLM not configured: semantic validation will remain NEEDS_ATTENTION."
    ),
    "ui.status.taxonomy": "Taxonomy",
    "ui.status.courses": "Courses: {courses}",
    "ui.status.prerequisites": "Prerequisites: {prerequisites}",
    "ui.review.session": "Staging session: `{session_id}`",
    "ui.review.warnings": "OCR warnings",
    "ui.review.note": (
        "The taxonomy, difficulty and prerequisites were inferred automatically during "
        "structuring; review and edit them before confirming."
    ),
    "ui.review.confirm_button": "Confirm and persist approved questions",
    "ui.review.nothing_approved": "There are no approved questions to save.",
    "ui.review.invalid_slugs": (
        "Course and topic must be valid slugs (lowercase, digits and underscore)."
    ),
    "ui.review.taxonomy_error": "Could not update the taxonomy: {error}",
    "ui.review.saved": "Persisted {count} question(s) to taygeta.db.",
    "ui.dark_mode.tooltip": "Switch between light and dark mode",
    # -- Question panel -----------------------------------------------------
    "ui.panel.header": "Question {number} — {status}",
    "ui.panel.valid": "Validation passed (VALID). Review it and approve it to save.",
    "ui.panel.attention": "NEEDS_ATTENTION: {reasons}",
    "ui.panel.rejection": "Semantic rejection reason: {reason}",
    "ui.panel.no_statement": "There is no structured statement to edit for this question.",
    "ui.panel.tab.original": "Original image",
    "ui.panel.tab.figures": "New figures",
    "ui.panel.approve": "Approve and save this question",
    "ui.panel.statement": "Statement (Markdown/TeX)",
    "ui.panel.qtype": "Question type",
    "ui.panel.notation_notes": "Notation notes: {notes}",
    "ui.panel.metadata": "Metadata",
    "ui.panel.choice.id": "Choice {number} — id",
    "ui.panel.choice.content": "Choice {number} — content",
    "ui.panel.course": "Course",
    "ui.panel.course.tooltip": "Course from the catalog or type a new one manually.",
    "ui.panel.topic": "Topic",
    "ui.panel.topic.tooltip": "Topics of the selected course or type a new one manually.",
    "ui.panel.subtopic": "Subtopic (snake_case slug, optional)",
    "ui.panel.difficulty": "Difficulty: {value}",
    "ui.panel.prerequisites": "Prerequisites",
    "ui.panel.advanced": "Advanced options",
    "ui.panel.hint": "Corrective context (e.g. «The illegible condition is g(x) = ...»)",
    "ui.panel.refine": "Re-analyze statement",
    "ui.panel.new_prereqs": "New prerequisites (comma-separated)",
    "ui.panel.dangling": (
        "The statement references placeholders with no declared figure: {placeholders}. "
        "Remove the placeholder or restore the figure to approve."
    ),
    "ui.panel.figures_title": "Question figures",
    "ui.panel.figure.load_failed": "{figure_id}: could not load the crop (page {page})",
    "ui.panel.figure.caption": "{figure_id} — crop (page {page})",
    "ui.panel.figure.open": "Open {figure_id} in the browser",
    "ui.panel.figure.delete": "Delete figure",
    "ui.panel.figure.recrop": "Crop manually — {figure_id}",
    "ui.panel.figure.recrop_none": "There is no source document to crop this figure.",
    "ui.panel.figure.recrop_caption": "Drag on page {page} to adjust the crop",
    "ui.panel.figure.apply_crop": "Apply crop",
    "ui.panel.figure.select_region_warning": ("Drag on the page to select the region"),
    "ui.panel.figure.crop_failed": "Could not apply the crop for {figure_id}: {error}",
    "ui.panel.figure.cropped": "{figure_id} cropped and saved.",
    "ui.panel.figure.add_expansion": "➕ Add new figure",
    "ui.panel.figure.no_source_add": "There is no source document to add figures.",
    "ui.panel.figure.page": "Page {number}",
    "ui.panel.figure.page_label": "Source page of the document",
    "ui.panel.figure.page_failed": "Could not load the source page to crop.",
    "ui.panel.figure.add_button": "Add figure",
    "ui.panel.figure.add_failed": "Could not add the figure: {error}",
    "ui.panel.figure.added": (
        "{figure_id} added; its placeholder was placed at the end of the statement."
    ),
    "ui.panel.figure.delete_failed": "Could not delete the figure: {error}",
    "ui.panel.original.none": "There is no source document to display.",
    "ui.panel.original.page": "Source page {number}",
    "ui.panel.original.page_failed": "Could not rasterize page {number}.",
    "ui.panel.refine_failed": "Could not re-analyze: {error}",
    # -- Figure cropper -----------------------------------------------------
    "ui.cropper.select_region": "Drag to select the region",
    "ui.cropper.select_figure": "Drag and drop to select the figure",
    "ui.cropper.zoom_reset": "Fit to view",
    # -- Persistence (surfaced as notifications) ----------------------------
    "ui.persist.figure_missing_asset": "{figure_id} (no asset_path)",
    "ui.persist.figure_not_in_bucket": (
        "{question_id}: figures not present in the bucket: {figures}"
    ),
    "ui.persist.session_update_failed": "could not mark the session as confirmed: {error}",
    # -- Generated exam documents -------------------------------------------
    "exam.title": "Practice exam",
    "exam.title_with_course": "Practice exam — {course}",
    "exam.question_heading": "Question {index}",
    "exam.latex.babel_language": "english",
    "exam.readme.latex": (
        "Exam source in LaTeX.\n\n"
        "Compile with: pdflatex main.tex\n\n"
        "The figures used are in the assets/ folder.\n"
    ),
    "exam.readme.typst": (
        "Exam source in Typst.\n\n"
        "Compile with: typst compile main.typ output.pdf\n\n"
        "The figures used are in the assets/ folder.\n"
    ),
}
