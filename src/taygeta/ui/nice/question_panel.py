"""Review panel for a single question in the NiceGUI UI.

Event-driven state model: the page is built once per session and events mutate
the state (``QuestionState``) and refresh the containers, with no reruns.

The panel contains:

* Markdown/TeX editor with a linked preview that renders the formulas with
  KaTeX (``ui.html`` + ``math_render``), not with ``ui.markdown`` (which does
  not support TeX and moreover destroys it with markdown2);
* question type and choices editor (only ``multiple_choice``);
* taxonomy with native course/topic comboboxes (``ui.select`` with
  ``new_value_mode="add-unique"``: pick from the catalog or type a new slug),
  subtopic, difficulty and prerequisites;
* advanced options: corrective context and statement re-analysis (``refine``)
  that rewrites the panel controls with the LLM response;
* figures on the right with "Original image" / "New figures" tabs, deletion,
  manual crop and addition of false negatives with the drag cropper; the
  displayed page numbering starts at 1 (the internal index is 0);
* approval checkbox that blocks persistence if the statement leaves dangling
  placeholders.

Visible text is resolved with :func:`taygeta.i18n.t` (``es``/``en`` catalogs).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from nicegui import run, ui
from PIL import Image

from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    TranscriptionIntermediate,
)
from taygeta.core.settings import Settings
from taygeta.i18n import t
from taygeta.ingest.pipeline import IngestPipeline, StagedQuestion
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database
from taygeta.server.object_storage import ObjectStorage
from taygeta.ui.figure_editing import (
    bbox_to_pixel_box,
    pixel_box_to_normalized_bbox,
    reconcile_figures_with_statement,
    remove_figure,
)
from taygeta.ui.nice import caption
from taygeta.ui.nice.figures import (
    DragBoxCropper,
    crop_new_figure,
    figure_pil,
    page_pil,
    replace_figure_crop,
)
from taygeta.ui.nice.math_render import MATH_CLASS, render_statement_html
from taygeta.ui.nice.state import QuestionState, SessionState
from taygeta.ui.shared import (
    cache_busted_public_url,
    document_page_count,
    page_image_bytes,
    slug_split,
)

QUESTION_TYPES = ["multiple_choice", "open_ended", "true_false", "numerical"]


def _taxonomy_defaults(staged: StagedQuestion) -> dict[str, Any]:
    """Safe default values (via ``getattr``) for the taxonomy.

    The staged object is an :class:`EnrichedQuestion` (the structurer infers the
    taxonomy in the single pass), so ``course``, ``topic``, ``subtopic``,
    ``difficulty`` and ``prerequisites`` already come filled in. It is accessed
    with ``getattr`` for robustness against unstructured questions
    (``question=None``).
    """
    base = staged.question
    course = getattr(base, "course", "") or ""
    topic = getattr(base, "topic", "") or ""
    subtopic = getattr(base, "subtopic", None)
    difficulty = getattr(base, "difficulty", 3)
    prerequisites = getattr(base, "prerequisites", None) or []

    if not isinstance(subtopic, str) or not subtopic:
        subtopic = None
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
        difficulty = 3
    return {
        "course": course,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "prerequisites": list(prerequisites),
    }


def _register_orphan(
    session: SessionState, question_id: str | None, asset_path: str | None
) -> None:
    """Register a figure asset as orphan to purge it on confirmation."""
    if not asset_path:
        return
    session.orphans.setdefault(str(question_id or ""), set()).add(asset_path)


@dataclass(slots=True)
class PanelDraft:
    """Collection of the panel values at confirmation time."""

    base: EnrichedQuestion
    statement: str
    qtype: str
    choices: list[Choice] | None
    course: str
    topic: str
    subtopic: str | None
    difficulty: int
    prerequisites: list[str]
    new_prereqs: list[str]
    effective_figures: list[dict[str, Any]]
    dangling: list[str]
    approved: bool


class QuestionPanel:
    """Review panel for a single question (editor, taxonomy, figures, approval)."""

    def __init__(
        self,
        parent: ui.element,
        *,
        index: int,
        staged: StagedQuestion,
        session: SessionState,
        source_path: str | None,
        storage: ObjectStorage | None,
        settings: Settings,
        database: Database,
        pipeline: IngestPipeline,
    ) -> None:
        self.index = index
        self.staged = staged
        self.base = staged.question
        self.session = session
        self.source_path = source_path
        self.storage = storage
        self.settings = settings
        self.database = database
        self.pipeline = pipeline
        self.qstate = session.questions.get(index) or QuestionState()
        if self.base is not None:
            self.qstate.figures = [
                figure if isinstance(figure, dict) else figure.model_dump(mode="json")
                for figure in (self.base.figures or [])
            ]
            session.questions[index] = self.qstate
        self.view = "Original page"
        self._dangling: list[str] = []

        # References to the widgets (to confirm and to rewrite after a re-analysis).
        self.statement_area: ui.textarea | None = None
        self.preview: ui.html | None = None
        self.qtype_select: ui.select | None = None
        self.choices_area: ui.column | None = None
        self.choice_editors: list[tuple[ui.input, ui.textarea]] = []
        self.course_select: ui.select | None = None
        self.topic_select: ui.select | None = None
        self.subtopic_input: ui.input | None = None
        self.difficulty_slider: ui.slider | None = None
        self.prereq_select: ui.select | None = None
        self.hint_input: ui.input | None = None
        self.new_prereq_input: ui.input | None = None
        self.approve_box: ui.checkbox | None = None
        self.dangling_label: ui.label | None = None
        self._refine_error: ui.label | None = None
        self._refine_spinner: ui.spinner | None = None
        self.right_body: ui.column | None = None
        self.figures_area: ui.column | None = None

        self._build(parent)

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def _build(self, parent: ui.element) -> None:
        staged = self.staged
        base = self.base
        with (
            parent,
            ui.expansion(
                t("ui.panel.header", number=self.index + 1, status=staged.status),
                value=staged.status == STATUS_NEEDS_ATTENTION,
            ) as expansion,
        ):
            # The editor uses ``autogrow`` and, with the panel collapsed, Quasar
            # measures its height as a single row without recalculating it when
            # expanded; the recalculation must be forced on open (see
            # ``_on_expansion_change``).
            expansion.on_value_change(self._on_expansion_change)
            if staged.status == STATUS_VALID:
                ui.label(t("ui.panel.valid")).classes("text-emerald-700")
            else:
                ui.label(t("ui.panel.attention", reasons="; ".join(staged.reasons))).classes(
                    "text-red-600"
                )
            if staged.logical is not None and not staged.logical.is_valid:
                ui.label(t("ui.panel.rejection", reason=staged.logical.rejection_reason)).classes(
                    "text-red-600"
                )
            if base is None:
                ui.label(t("ui.panel.no_statement"))
                return
            # A grid (not a row) is used because ``ui.row`` is
            # ``flex-wrap: wrap`` and gapped columns would stack once they
            # exceed 100% width.
            with ui.grid().classes("w-full gap-4 grid-cols-1 lg:grid-cols-2 items-start"):
                with ui.column().classes("w-full col-span-1 gap-3 min-w-0"):
                    self._build_editor()
                    self._build_taxonomy()
                    self._build_advanced()
                    self.dangling_label = ui.label("").classes("text-red-600 text-sm")
                    self.dangling_label.set_visibility(False)
                    self.approve_box = ui.checkbox(
                        t("ui.panel.approve"), value=staged.status == STATUS_VALID
                    )
                    self._sync_status()
                with ui.column().classes("w-full col-span-1 gap-3 min-w-0"):
                    with ui.tabs(
                        value="Original page",
                        on_change=lambda e: self._set_view(e.value),
                    ).classes("w-full"):
                        ui.tab("Original page", label=t("ui.panel.tab.original"))
                        ui.tab("Figures", label=t("ui.panel.tab.figures"))
                    self.right_body = ui.column().classes("w-full gap-2")
                    self._set_view("Original page")

    def _build_editor(self) -> None:
        base = self.base
        if base is None:
            return
        self.statement_area = ui.textarea(
            t("ui.panel.statement"), value=base.statement_raw
        ).classes("w-full")
        self.statement_area.props("outlined autogrow")
        self.preview = ui.html(render_statement_html(base.statement_raw)).classes(
            f"w-full border rounded p-2 {MATH_CLASS}"
        )
        self.statement_area.on_value_change(self._on_statement_change)

        self.qtype_select = ui.select(
            QUESTION_TYPES,
            label=t("ui.panel.qtype"),
            value=base.question_type,
            with_input=True,
        )
        self.qtype_select.on_value_change(lambda _e: self._rebuild_choices())
        self.choices_area = ui.column().classes("w-full gap-2")
        self._rebuild_choices()
        if base.notation_notes:
            ui.label(t("ui.panel.notation_notes", notes="; ".join(base.notation_notes))).classes(
                "text-sm text-gray-600"
            )
        ui.separator()
        ui.label(t("ui.panel.metadata")).classes("font-semibold")

    def _rebuild_choices(self) -> None:
        """(Re)build the choice editors according to the selected type."""
        if self.choices_area is None or self.qtype_select is None:
            return
        self.choices_area.clear()
        self.choice_editors = []
        if self.qtype_select.value != "multiple_choice":
            return
        base_choices = self.base.choices if self.base is not None else []
        with self.choices_area:
            for choice_index, choice in enumerate(base_choices):
                with ui.row().classes("w-full gap-2 items-start"):
                    choice_id = ui.input(
                        t("ui.panel.choice.id", number=choice_index + 1), value=choice.id
                    ).classes("w-32")
                    choice_text = ui.textarea(
                        t("ui.panel.choice.content", number=choice_index + 1),
                        value=choice.content_raw,
                    ).classes("flex-1")
                self.choice_editors.append((choice_id, choice_text))

    def _combobox(self, label: str, options: list[str], value: str) -> ui.select:
        """Slug combobox: dropdown catalog + entry of a new value.

        The current value is always included in the options so it does not break
        when the catalog is still empty or the inferred value is not cataloged.
        """
        merged = sorted({*options, value}) if value else list(options)
        return ui.select(
            merged,
            label=label,
            value=value or None,
            with_input=True,
            new_value_mode="add-unique",
        ).classes("w-full")

    def _topics_for(self, course: str) -> list[str]:
        if not course:
            return []
        return sorted({row["topic_slug"] for row in self.database.list_topics(course)})

    def _build_taxonomy(self) -> None:
        defaults = _taxonomy_defaults(self.staged)
        courses = sorted({row["slug"] for row in self.database.list_courses()})
        self.course_select = self._combobox(
            t("ui.panel.course"), courses, defaults["course"]
        ).tooltip(t("ui.panel.course.tooltip"))
        self.topic_select = self._combobox(
            t("ui.panel.topic"), self._topics_for(defaults["course"]), defaults["topic"]
        ).tooltip(t("ui.panel.topic.tooltip"))
        self.course_select.on_value_change(self._on_course_change)

        self.subtopic_input = ui.input(t("ui.panel.subtopic"), value=defaults["subtopic"] or "")
        with ui.row().classes("items-center gap-4"):
            self.difficulty_slider = ui.slider(
                min=1, max=5, step=1, value=int(defaults["difficulty"])
            )
            ui.label().bind_text_from(
                self.difficulty_slider,
                "value",
                backward=lambda v: t("ui.panel.difficulty", value=v),
            )
        catalog = sorted({row["slug"] for row in self.database.list_prerequisites()})
        prereq_default = list(dict.fromkeys(defaults["prerequisites"]))
        options = sorted({*catalog, *prereq_default})
        self.prereq_select = ui.select(
            options,
            label=t("ui.panel.prerequisites"),
            value=prereq_default,
            multiple=True,
            with_input=True,
            new_value_mode="add-unique",
        ).classes("w-full")

    def _build_advanced(self) -> None:
        with ui.expansion(t("ui.panel.advanced")).classes("w-full"):
            self.hint_input = ui.input(t("ui.panel.hint")).classes("w-full")
            self._refine_error = ui.label("").classes("text-red-600 text-sm")
            self._refine_error.set_visibility(False)
            with ui.row().classes("items-center gap-2"):
                ui.button(t("ui.panel.refine"), on_click=self._refine)
                self._refine_spinner = ui.spinner(size="sm")
                self._refine_spinner.set_visibility(False)
            self.new_prereq_input = ui.input(t("ui.panel.new_prereqs")).classes("w-full")

    # ------------------------------------------------------------------
    # Editor and taxonomy events
    # ------------------------------------------------------------------

    def _render_preview(self, statement: str) -> None:
        """Re-render the statement preview (Markdown + TeX/KaTeX)."""
        if self.preview is not None:
            self.preview.set_content(render_statement_html(statement or ""))

    def _on_statement_change(self, e: Any) -> None:
        value = getattr(e, "value", None)
        if value is None and self.statement_area is not None:
            value = self.statement_area.value
        self._render_preview(value or "")
        self._sync_status()

    def _on_expansion_change(self, e: Any) -> None:
        """Re-adjust the editor height when expanding the question panel.

        The statement is edited in a ``ui.textarea`` with ``autogrow`` inside a
        ``ui.expansion`` that starts collapsed. Quasar measures the ``autogrow``
        height when mounting the component; with the panel hidden it computes it
        as a single row and does not recalculate it when expanded, so the
        statement is clipped to its first line (with hidden ``overflow``). When
        opening the panel we dispatch an ``input`` event on the textarea so
        Quasar re-measures its content.
        """
        if not getattr(e, "value", False) or self.statement_area is None:
            return
        ui.run_javascript(
            f"const el = document.getElementById('{self.statement_area.html_id}');"
            "if (el) el.dispatchEvent(new Event('input', { bubbles: true }));"
        )

    def _on_course_change(self, e: Any) -> None:
        course = e.value or ""
        topics = self._topics_for(course)
        if self.topic_select is not None:
            self.topic_select.options = topics
            if self.topic_select.value not in topics:
                self.topic_select.value = None
            self.topic_select.update()

    def _sync_status(self) -> None:
        """Reconcile figures with the statement and update the dangling warning."""
        if self.statement_area is None:
            return
        statement = self.statement_area.value or ""
        _effective, orphans, dangling = reconcile_figures_with_statement(
            self.qstate.figures, statement
        )
        for figure in orphans:
            _register_orphan(
                self.session, str(self.base.id) if self.base else None, figure.get("asset_path")
            )
        self._dangling = dangling
        if self.dangling_label is None:
            return
        if dangling:
            self.dangling_label.set_text(
                t("ui.panel.dangling", placeholders=", ".join(sorted(dangling)))
            )
            self.dangling_label.set_visibility(True)
        else:
            self.dangling_label.set_visibility(False)

    # ------------------------------------------------------------------
    # Figures (right column)
    # ------------------------------------------------------------------

    def _set_view(self, view: str) -> None:
        self.view = view
        if self.right_body is None:
            return
        self.right_body.clear()
        # When leaving "Figures" we invalidate the reference so later
        # updates (refine, deletion) do not try to redraw over an
        # already-destroyed view; it is rebuilt when returning to the tab.
        self.figures_area = None
        with self.right_body:
            if view == "Figures":
                self.figures_area = ui.column().classes("w-full gap-2")
                self._render_figures()
            else:
                self._render_original_pages()

    def _render_figures(self) -> None:
        """Draw (or redraw) the list of figures and the manual addition."""
        if self.figures_area is None:
            return
        self.figures_area.clear()
        with self.figures_area:
            ui.label(t("ui.panel.figures_title")).classes("font-semibold")
            for figure in self.qstate.figures:
                figure_id = figure.get("id") or "fig:?"
                page_index = int(figure.get("page", 0))
                with ui.card().classes("w-full gap-1"):
                    image = figure_pil(self.storage, figure.get("asset_path"))
                    if image is not None:
                        ui.image(image).classes("w-full rounded border")
                    else:
                        caption(
                            t(
                                "ui.panel.figure.load_failed",
                                figure_id=figure_id,
                                page=page_index + 1,
                            )
                        )
                    caption(
                        t(
                            "ui.panel.figure.caption",
                            figure_id=figure_id,
                            page=page_index + 1,
                        )
                    )
                    url = cache_busted_public_url(
                        figure.get("asset_path"), self.settings, seed=figure.get("bbox")
                    )
                    if url:
                        ui.link(
                            t("ui.panel.figure.open", figure_id=figure_id),
                            url,
                            new_tab=True,
                        ).classes("text-xs")
                    ui.button(
                        t("ui.panel.figure.delete"),
                        on_click=lambda fid=figure_id: self._delete_figure(fid),
                    ).props("flat color=negative dense")
                    with ui.expansion(t("ui.panel.figure.recrop", figure_id=figure_id)):
                        self._render_recrop(figure_id, figure, page_index)
            self._render_add_area()

    def _render_recrop(self, figure_id: str, figure: dict[str, Any], page_index: int) -> None:
        result = page_pil(self.source_path, page_index)
        if result is None:
            caption(t("ui.panel.figure.recrop_none"))
            return
        image, width, height = result
        pending: dict[str, Any] = {"bbox": None}
        DragBoxCropper(
            image,
            width=width,
            height=height,
            on_confirm=lambda box: pending.update({"bbox": box}),
            caption=t("ui.panel.figure.recrop_caption", page=page_index + 1),
            initial_box=self._initial_pixel_box(figure, width, height),
        )
        ui.button(
            t("ui.panel.figure.apply_crop"),
            on_click=self._apply_recrop(figure_id, width, height, pending),
        ).props("dense")

    @staticmethod
    def _initial_pixel_box(
        figure: dict[str, Any], width: int, height: int
    ) -> dict[str, int] | None:
        """Pixel box of the figure's current BBox, to seed the cropper.

        Returns the current box ``{"left", "top", "width", "height"}`` or
        ``None`` if the figure has no valid BBox for the given dimensions.
        """
        bbox = figure.get("bbox")
        if not bbox:
            return None
        try:
            xl, xr, yt, yb = bbox_to_pixel_box(bbox, int(width), int(height))
        except (ValueError, TypeError):
            return None
        return {"left": xl, "top": yt, "width": xr - xl, "height": yb - yt}

    def _apply_recrop(
        self, figure_id: str, width: int, height: int, pending: dict[str, Any]
    ) -> Any:
        async def handler() -> None:
            box = pending.get("bbox")
            if box is None:
                ui.notify(t("ui.panel.figure.select_region_warning"), type="warning")
                return
            try:
                new_bbox = pixel_box_to_normalized_bbox(box, int(width), int(height))
                figures = await run.io_bound(
                    replace_figure_crop,
                    self.qstate.figures,
                    figure_id,
                    new_bbox,
                    self.source_path,
                    self.storage,
                )
            except Exception as exc:  # noqa: BLE001 - reported to the user, not hidden
                ui.notify(
                    t("ui.panel.figure.crop_failed", figure_id=figure_id, error=exc),
                    type="negative",
                )
                return
            self.qstate.figures = figures
            ui.notify(t("ui.panel.figure.cropped", figure_id=figure_id), type="positive")
            self._render_figures()

        return handler

    def _render_add_area(self) -> None:
        with ui.expansion(t("ui.panel.figure.add_expansion")) as add_exp:
            page_count = document_page_count(self.source_path)
            if page_count is None or page_count <= 0:
                caption(t("ui.panel.figure.no_source_add"))
                return
            default_page = 0
            if self.qstate.figures:
                default_page = min(int(self.qstate.figures[0].get("page", 0)), int(page_count) - 1)
            page_select = ui.select(
                {
                    index: t("ui.panel.figure.page", number=index + 1)
                    for index in range(int(page_count))
                },
                value=default_page,
                label=t("ui.panel.figure.page_label"),
            ).classes("w-full")
            body = ui.column().classes("w-full gap-2")
            pending: dict[str, Any] = {"bbox": None, "page": int(default_page)}
            dims: dict[str, int] = {}

            def show_cropper() -> None:
                body.clear()
                with body:
                    page_index = int(page_select.value)
                    result = page_pil(self.source_path, page_index)
                    if result is None:
                        caption(t("ui.panel.figure.page_failed"))
                        return
                    image, width, height = result
                    dims["width"], dims["height"] = width, height
                    pending.clear()
                    pending["page"] = page_index
                    pending["bbox"] = None
                    DragBoxCropper(
                        image,
                        width=width,
                        height=height,
                        on_confirm=lambda box: pending.update({"bbox": box}),
                        caption=t("ui.cropper.select_figure"),
                    )
                    ui.button(
                        t("ui.panel.figure.add_button"),
                        on_click=self._add_figure(pending, dims),
                    ).props("dense")

            page_select.on_value_change(lambda _e: show_cropper())
            # The cropper is drawn when the expansion is opened (not at build
            # time) so as not to rasterize a page for every panel while building
            # the review.
            add_exp.on_value_change(lambda e: show_cropper() if e.value else None)

    def _add_figure(self, pending: dict[str, Any], dims: dict[str, int]) -> Any:
        async def handler() -> None:
            box = pending.get("bbox")
            if box is None or self.statement_area is None:
                ui.notify(t("ui.panel.figure.select_region_warning"), type="warning")
                return
            try:
                new_bbox = pixel_box_to_normalized_bbox(box, dims["width"], dims["height"])
                figures, statement, figure_id = await run.io_bound(
                    crop_new_figure,
                    self.qstate.figures,
                    self.statement_area.value,
                    page_index=int(pending["page"]),
                    bbox=list(new_bbox),
                    source_path=self.source_path,
                    storage=self.storage,
                )
            except Exception as exc:  # noqa: BLE001 - reported to the user, not hidden
                ui.notify(t("ui.panel.figure.add_failed", error=exc), type="negative")
                return
            self.qstate.figures = figures
            self.statement_area.value = statement
            self._render_preview(statement)
            self._sync_status()
            ui.notify(t("ui.panel.figure.added", figure_id=figure_id), type="positive")
            self._render_figures()

        return handler

    def _delete_figure(self, figure_id: str) -> None:
        if self.statement_area is None:
            return
        try:
            figures, statement, removed = remove_figure(
                self.qstate.figures, self.statement_area.value, figure_id
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user, not hidden
            ui.notify(t("ui.panel.figure.delete_failed", error=exc), type="negative")
            return
        self.qstate.figures = figures
        for figure in removed:
            _register_orphan(
                self.session,
                str(self.base.id) if self.base else None,
                figure.get("asset_path"),
            )
        self.statement_area.value = statement
        self._render_preview(statement)
        self._sync_status()
        self._render_figures()

    def _render_original_pages(self) -> None:
        if self.source_path is None:
            caption(t("ui.panel.original.none"))
            return
        pages = [0]
        if self.base is not None and self.base.figures:
            pages = sorted({figure.page for figure in self.base.figures})
        for page_index in pages:
            data = page_image_bytes(self.source_path, page_index)
            if data:
                ui.image(Image.open(io.BytesIO(data))).classes("w-full rounded border")
                caption(t("ui.panel.original.page", number=page_index + 1))
            else:
                caption(t("ui.panel.original.page_failed", number=page_index + 1))

    # ------------------------------------------------------------------
    # Statement re-analysis (refine)
    # ------------------------------------------------------------------

    async def _refine(self) -> None:
        result = self.session.staging
        if result is None or self.base is None:
            return
        hint = self.hint_input.value or None if self.hint_input is not None else None
        intermediate = result.intermediate
        if intermediate is None:
            intermediate = TranscriptionIntermediate(
                source_reference=result.source_reference,
                raw_markdown=self.base.statement_raw,
                pages=[],
                figures=self.base.figures,
                warnings=[],
            )
        if self._refine_error is not None:
            self._refine_error.set_visibility(False)
        if self._refine_spinner is not None:
            self._refine_spinner.set_visibility(True)
        try:
            refined = await run.io_bound(
                self.pipeline.refine_question, intermediate, self.base, hints=hint
            )
        except Exception as exc:  # noqa: BLE001 - reported to the panel, not hidden
            if self._refine_error is not None:
                self._refine_error.set_text(t("ui.panel.refine_failed", error=exc))
                self._refine_error.set_visibility(True)
            return
        finally:
            if self._refine_spinner is not None:
                self._refine_spinner.set_visibility(False)
        result.questions[self.index] = refined
        self._apply_refine(refined)

    def _apply_refine(self, staged: StagedQuestion) -> None:
        """Rewrite the panel controls with the re-analysis result."""
        question = staged.question
        if question is None:
            return
        self.staged = staged
        self.base = question
        self.qstate.figures = [
            figure if isinstance(figure, dict) else figure.model_dump(mode="json")
            for figure in (question.figures or [])
        ]
        if self.statement_area is not None:
            self.statement_area.value = question.statement_raw
        self._render_preview(question.statement_raw)
        if self.qtype_select is not None:
            self.qtype_select.value = question.question_type
        self._rebuild_choices()
        if self.course_select is not None:
            self.course_select.value = question.course
        if self.topic_select is not None:
            topics = self._topics_for(question.course)
            self.topic_select.options = topics
            self.topic_select.value = question.topic or None
            self.topic_select.update()
        if self.subtopic_input is not None:
            self.subtopic_input.value = question.subtopic or ""
        if self.difficulty_slider is not None:
            self.difficulty_slider.value = int(question.difficulty)
        if self.prereq_select is not None:
            options = sorted({*self.prereq_select.options, *question.prerequisites})
            self.prereq_select.options = options
            self.prereq_select.value = list(dict.fromkeys(question.prerequisites))
            self.prereq_select.update()
        self._sync_status()
        self._render_figures()

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def collect(self) -> PanelDraft | None:
        """Collect the panel's current values (or ``None`` if there is no statement)."""
        base = self.base
        if base is None or self.statement_area is None or self.qtype_select is None:
            return None
        statement = self.statement_area.value or ""
        qtype = str(self.qtype_select.value or base.question_type)
        choices: list[Choice] | None = None
        if qtype == "multiple_choice":
            choices = [
                Choice(id=choice_id.value.strip(), content_raw=choice_text.value.strip())
                for choice_id, choice_text in self.choice_editors
            ]
        effective, orphans, dangling = reconcile_figures_with_statement(
            self.qstate.figures, statement
        )
        for figure in orphans:
            _register_orphan(self.session, str(base.id), figure.get("asset_path"))
        course = str(self.course_select.value or "") if self.course_select is not None else ""
        topic = str(self.topic_select.value or "") if self.topic_select is not None else ""
        subtopic = self.subtopic_input.value if self.subtopic_input is not None else ""
        difficulty = (
            int(self.difficulty_slider.value)
            if self.difficulty_slider is not None and self.difficulty_slider.value
            else 3
        )
        prerequisites = (
            list(self.prereq_select.value or []) if self.prereq_select is not None else []
        )
        new_prereqs = (
            slug_split(self.new_prereq_input.value) if self.new_prereq_input is not None else []
        )
        approved = bool(self.approve_box.value) if self.approve_box is not None else False
        return PanelDraft(
            base=base,
            statement=statement,
            qtype=qtype,
            choices=choices,
            course=course,
            topic=topic,
            subtopic=slug_split(subtopic)[0] if subtopic else None,
            difficulty=difficulty,
            prerequisites=prerequisites,
            new_prereqs=new_prereqs,
            effective_figures=effective,
            dangling=dangling,
            approved=approved,
        )
