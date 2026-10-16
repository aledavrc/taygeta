"""Tests for the NiceGUI port of the HITL review (``user_simulation`` harness).

These tests build the real page without a browser (in-process ``httpx`` +
``socketio``). Important notes:

* ``user_simulation`` **only works under pytest**: the globals reset calls
  ``App.reset()`` -> ``Storage.clear()``, which accesses ``context.client``
  unless ``helpers.is_pytest()`` is true. Running it as a standalone script
  enables "script mode" and breaks the ``@ui.page``-based app.
* Playwright is not needed (``screen``/Selenium would be).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from nicegui import ui

from taygeta.core.schema import Choice, EnrichedQuestion, FigureReference, utc_now_iso
from taygeta.core.settings import get_settings
from taygeta.ingest.pipeline import StagedQuestion
from taygeta.ingest.validator import STATUS_NEEDS_ATTENTION, STATUS_VALID
from taygeta.server.db import Database
from taygeta.ui.nice.math_render import MATH_CLASS
from taygeta.ui.nice.question_panel import QuestionPanel
from taygeta.ui.nice.state import QuestionState, SessionState

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Force the anyio plugin's asyncio backend."""
    return "asyncio"


@pytest.fixture
def ui_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate UI paths/persistence in a temporary directory."""
    for name in ("uploads", "downloads"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "taygeta.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path / "downloads"))
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:8333")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _StubStorage:
    """Null storage: avoids any network access when loading figures."""

    bucket = "taygeta-assets"

    def get_bytes(self, key: str, bucket: str | None = None) -> bytes | None:
        return None


class _FakeStorage:
    """In-memory storage to test figure adding/cropping without the network."""

    bucket = "taygeta-assets"

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_bytes(self, key: str, bucket: str | None = None) -> bytes | None:
        return self.objects.get(key)

    def store_asset_bytes(
        self,
        data: bytes,
        *,
        prefix: str = "figures",
        asset_id: str | None = None,
        extension: str = ".png",
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        key = f"{prefix}/{asset_id or uuid4().hex}{extension}"
        self.objects[key] = data
        return f"s3://{self.bucket}/{key}"

    def relative_path(self, reference: str) -> str:
        key = reference.split("/", 3)[3]
        return f"/buckets/{self.bucket}/{key}"


def _make_questions() -> list[StagedQuestion]:
    valid = EnrichedQuestion(
        id=str(uuid4()),
        statement_raw="Resuelve $x^2 - 1 = 0$ y elige la alternativa. {{fig:1}}",
        question_type="multiple_choice",
        choices=[Choice(id="A", content_raw="$x=1$"), Choice(id="B", content_raw="$x=-1$")],
        figures=[
            FigureReference(
                id="fig:1",
                page=0,
                bbox=[100, 100, 400, 400],
                asset_path="/buckets/taygeta-assets/figures/x.png",
            )
        ],
        course="algebra",
        topic="ecuaciones",
        subtopic="cuadraticas",
        difficulty=2,
        prerequisites=["factorizacion"],
        created_at=utc_now_iso(),
    )
    attention = EnrichedQuestion(
        id=str(uuid4()),
        statement_raw="Demuestra que la suma de ángulos internos de un triángulo es 180°.",
        question_type="open_ended",
        choices=None,
        course="geometria",
        topic="triangulos",
        subtopic=None,
        difficulty=3,
        prerequisites=[],
        created_at=utc_now_iso(),
    )
    return [
        StagedQuestion(question=valid, status=STATUS_VALID, reasons=[]),
        StagedQuestion(
            question=attention,
            status=STATUS_NEEDS_ATTENTION,
            reasons=["validación semántica incompleta"],
        ),
    ]


async def test_index_page_builds(ui_env: None, tmp_path: Path) -> None:
    """The NiceGUI port's main page builds without errors."""
    from nicegui.testing import user_simulation

    launcher = tmp_path / "niceui_launcher.py"
    launcher.write_text("from taygeta.ui.nice.app import main\n\nmain()\n")

    async with user_simulation(main_file=launcher) as user:
        await user.open("/")
        await user.should_see(content="Taygeta — Ingesta y revisión Human-in-the-Loop")
        await user.should_see("1. Subir documento")
        await user.should_see("2. Procesar")
        await user.should_see("3. Revisión y confirmación")

        # Dark mode: the floating button toggles the client state.
        user.find("toggle-dark-mode").click()
        dark = next(
            element
            for element in user.find(ui.dark_mode).elements
            if isinstance(element, ui.dark_mode)
        )
        assert dark.value is True


async def test_question_panels_build_and_switch_view(ui_env: None) -> None:
    """The panels build with fake data and respond to events."""
    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()

    def root() -> None:
        session = SessionState()
        session.questions = {index: QuestionState() for index in range(len(staged))}
        for index, item in enumerate(staged):
            QuestionPanel(
                ui.column().classes("w-full"),
                index=index,
                staged=item,
                session=session,
                source_path=None,
                storage=_StubStorage(),
                settings=settings,
                database=database,
                pipeline=None,
            )

    from nicegui.testing import user_simulation

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see(content="Pregunta 1 — VALID")
        await user.should_see(content="Pregunta 2 — NEEDS_ATTENTION")
        await user.should_see("Aprobar y guardar esta pregunta")
        await user.should_see("Alternativa 1 — id")
        await user.should_see("Opciones Avanzadas")

        # The right column's default view is "Original image".
        await user.should_see("Imagen original")
        await user.should_see("No hay documento de origen para mostrar.")

        # Switching to "New figures" shows figure management.
        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")


async def test_question_panels_render_in_english(ui_env: None) -> None:
    """The same panels render their strings in English when the locale is 'en'."""
    import taygeta.i18n as i18n
    from taygeta.ui.nice import localization

    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()

    def root() -> None:
        session = SessionState()
        session.questions = {index: QuestionState() for index in range(len(staged))}
        for index, item in enumerate(staged):
            QuestionPanel(
                ui.column().classes("w-full"),
                index=index,
                staged=item,
                session=session,
                source_path=None,
                storage=_StubStorage(),
                settings=settings,
                database=database,
                pipeline=None,
            )

    from nicegui.testing import user_simulation

    # Force the locale to English for the duration of the simulation, then
    # restore the session-backed resolver installed by the UI localization layer.
    i18n.set_locale_resolver(lambda: "en")
    try:
        async with user_simulation(root=root) as user:
            await user.open("/")
            await user.should_see(content="Question 1 — VALID")
            await user.should_see(content="Question 2 — NEEDS_ATTENTION")
            await user.should_see("Approve and save this question")
            await user.should_see("Choice 1 — id")
            await user.should_see("Advanced options")
            await user.should_see("Original image")
            await user.should_see("There is no source document to display.")

            # Switching to "New figures" shows figure management (English).
            user.find("New figures").click()
            await user.should_see("➕ Add new figure")
    finally:
        i18n.set_locale_resolver(localization.active_locale)


async def test_add_figure_area_renders_cropper_with_real_source(
    ui_env: None, tmp_path: Path
) -> None:
    """Regression: with a real source, adding figures draws the cropper.

    Previously, ``_render_add_area`` called ``Select.on_change`` (nonexistent in
    NiceGUI 3), which aborted the panel build and only showed the first question
    and an empty page selector.
    """
    from nicegui.testing import user_simulation
    from PIL import Image

    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()[:1]
    source = tmp_path / "pagina.png"
    Image.new("RGB", (800, 1000), "white").save(source)

    def root() -> None:
        session = SessionState()
        session.questions = {0: QuestionState()}
        QuestionPanel(
            ui.column().classes("w-full"),
            index=0,
            staged=staged[0],
            session=session,
            source_path=str(source),
            storage=_StubStorage(),
            settings=settings,
            database=database,
            pipeline=None,
        )

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("Pregunta 1 — VALID")

        # The displayed page numbering starts at 1.
        await user.should_see("Página original 1")
        await user.should_not_see("Página original 0")

        # Switch to the figures tab: figure adding used to be built here.
        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")

        # Expanding the add area draws the cropper (deferred render).
        add_expansion = next(
            element
            for element in user.find("➕ Añadir nueva figura").elements
            if isinstance(element, ui.expansion)
        )
        with user:
            add_expansion.value = True
        await user.should_see("Arrastra y suelta para seleccionar la figura")
        await user.should_see("Añadir figura")


def _cropper_root(source: Path, staged: StagedQuestion, storage: Any) -> Callable[[], None]:
    """Return a ``root`` with a single ``QuestionPanel`` for the simulation."""
    settings = get_settings()
    database = Database(settings=settings)

    def root() -> None:
        session = SessionState()
        session.questions = {0: QuestionState()}
        QuestionPanel(
            ui.column().classes("w-full"),
            index=0,
            staged=staged,
            session=session,
            source_path=str(source),
            storage=storage,
            settings=settings,
            database=database,
            pipeline=None,
        )

    return root


async def test_add_figure_keeps_cropper_selection(ui_env: None, tmp_path: Path) -> None:
    """Regression: the box is kept on release and the figure is added.

    Previously, ``DragBoxCropper`` cleared the rectangle on ``mouseup`` and
    ``on_confirm`` stored ``left/top/width/height`` instead of ``bbox``, so
    adding always warned that there was no selection.
    """
    from nicegui.testing import user_simulation
    from PIL import Image

    staged = _make_questions()[1]  # no figures: a single cropper in the panel
    source = tmp_path / "pagina.png"
    Image.new("RGB", (600, 800), "white").save(source)
    root = _cropper_root(source, staged, _FakeStorage())

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("Pregunta 1 — NEEDS_ATTENTION")

        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")
        add_expansion = next(
            element
            for element in user.find("➕ Añadir nueva figura").elements
            if isinstance(element, ui.expansion)
        )
        with user:
            add_expansion.value = True

        cropper_ui = user.find(ui.interactive_image)
        cropper = next(
            element for element in cropper_ui.elements if isinstance(element, ui.interactive_image)
        )

        def mouse(kind: str, x: int, y: int) -> None:
            cropper_ui.trigger("mouse", {"mouse_event_type": kind, "image_x": x, "image_y": y})

        mouse("mousedown", 50, 60)
        mouse("mousemove", 250, 260)
        mouse("mouseup", 250, 260)
        # The box is still drawn after releasing the mouse.
        assert "<rect" in cropper.content

        user.find("Añadir figura").click()
        await user.should_see("fig:1 — recorte (página 1)", retries=15)


async def test_cropper_zoom_controls(ui_env: None, tmp_path: Path) -> None:
    """The zoom controls zoom in, zoom out and reset the cropper.

    Zoom resizes the image through *layout* (not with ``transform``), so
    ``ui.interactive_image``'s coordinate mapping stays exact.
    """
    from nicegui.testing import user_simulation
    from PIL import Image

    staged = _make_questions()[1]  # no figures: a single cropper in the panel
    source = tmp_path / "pagina.png"
    Image.new("RGB", (600, 800), "white").save(source)
    root = _cropper_root(source, staged, _FakeStorage())

    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("Pregunta 1 — NEEDS_ATTENTION")

        user.find("Figuras nuevas").click()
        await user.should_see("➕ Añadir nueva figura")
        add_expansion = next(
            element
            for element in user.find("➕ Añadir nueva figura").elements
            if isinstance(element, ui.expansion)
        )
        with user:
            add_expansion.value = True

        # The cropper starts fitted to the view.
        await user.should_see("100 %")

        # Zooming in steps up by 0.5; reset returns to 100 %.
        user.find("cropper-zoom-in").click()
        await user.should_see("150 %")
        user.find("cropper-zoom-in").click()
        await user.should_see("200 %")
        user.find("cropper-zoom-reset").click()
        await user.should_see("100 %")

        # Zooming out below the minimum does not go under 100 % (the clamp prevents it).
        user.find("cropper-zoom-out").click()
        await user.should_see("100 %")


async def test_statement_preview_renders_math(ui_env: None) -> None:
    """The formatting preview is ``ui.html`` + KaTeX and preserves formulas.

    It used to use ``ui.markdown``: markdown2 mangled the TeX (``$x^2$`` →
    ``$x<em>2</em>$``) and no engine rendered it.
    """
    from nicegui.testing import user_simulation

    settings = get_settings()
    database = Database(settings=settings)
    staged = _make_questions()

    def root() -> None:
        session = SessionState()
        session.questions = {index: QuestionState() for index in range(len(staged))}
        QuestionPanel(
            ui.column().classes("w-full"),
            index=0,
            staged=staged[0],
            session=session,
            source_path=None,
            storage=_StubStorage(),
            settings=settings,
            database=database,
            pipeline=None,
        )

    async with user_simulation(root=root) as user:
        await user.open("/")
        previews = [
            element for element in user.find(ui.html).elements if MATH_CLASS in element.classes
        ]
        assert len(previews) == 1
        content = previews[0].content
        # The formula arrives intact (without the emphasis markdown2 introduced).
        assert "$x^2 - 1 = 0$" in content
        assert "<em>" not in content
        # The figure placeholder stays as text.
        assert "{{fig:1}}" in content
