"""Practice exam generation (RAG, Typst, ZIP) — section 9.3.

Responsibilities:

* ``retrieve_questions``: retrieves real exercises by course/topic/difficulty
  using parameterized SQLite queries (FTS5 + filters).
* ``build_exam_typst`` / ``build_exam_latex``: assemble the template from the
  statements (Markdown/TeX) through the transpiler.
* ``copy_assets``: downloads from SeaweedFS **only** the figures that are used
  and copies them into the ``assets/`` directory of the build workspace.
* ``generate_exam``: retrieves, assembles, copies assets and compiles the PDF
  with the native Typst CLI; returns a unique ``exam_id``, a summary and a
  temporary URL.
* ``build_source_zip``: packages ``main.typ`` (or ``main.tex``), ``assets/`` and
  ``README.txt`` into a ZIP for download.

No solutions or correct answers are generated or included.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taygeta.core import transpiler
from taygeta.core.schema import EnrichedQuestion, FigureReference
from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import split_asset_reference
from taygeta.domain.taxonomy import format_course_name
from taygeta.i18n import DEFAULT_LOCALE
from taygeta.server.db import Database
from taygeta.server.object_storage import ObjectStorage

logger = logging.getLogger(__name__)

__all__ = ["ExamGenerator", "ExamGenerationError", "ExamResult"]

_ALLOWED_FORMATS = frozenset({"typst", "latex"})
_HEX_CHARS = frozenset("0123456789abcdef")


class ExamGenerationError(RuntimeError):
    """Controlled error of exam generation or packaging."""


def _is_valid_exam_id(exam_id: str) -> bool:
    return len(exam_id) == 32 and all(char in _HEX_CHARS for char in exam_id)


@dataclass(slots=True)
class ExamResult:
    """Result of an exam generation."""

    exam_id: str
    status: str
    num_questions: int
    pdf_path: Path | None
    pdf_url: str | None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "exam_id": self.exam_id,
            "status": self.status,
            "num_questions": self.num_questions,
            "pdf_url": self.pdf_url,
            "warnings": self.warnings,
        }


class ExamGenerator:
    """Practice exam generation and packaging engine."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database: Database | None = None,
        storage: ObjectStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database = database or Database(settings=self.settings)
        self.storage = storage or ObjectStorage(self.settings)

    # -- Retrieval (RAG) ----------------------------------------------------
    def retrieve_questions(
        self,
        *,
        course: str | None = None,
        topics: Sequence[str] | None = None,
        difficulty: int | None = None,
        limit: int = 10,
    ) -> list[EnrichedQuestion]:
        """Retrieves questions filtered by course, topics and difficulty.

        Uses parameterized ``Database`` queries and a bounded limit. It does not
        return solutions or correct answers.
        """
        limit = max(1, min(limit, 100))
        if topics:
            seen: dict[str, EnrichedQuestion] = {}
            for topic in topics:
                for question in self.database.list_questions(
                    course=course, topic=topic, difficulty=difficulty, limit=limit
                ):
                    seen[question.id] = question
            return list(seen.values())[:limit]
        return self.database.list_questions(course=course, difficulty=difficulty, limit=limit)

    # -- Local figure naming ------------------------------------------------
    @staticmethod
    def local_asset_name(figure: FigureReference) -> str:
        """Safe local file name for a figure (relative to ``assets/``)."""
        suffix = Path(figure.asset_path or "").suffix or ".png"
        if not suffix.startswith("."):
            suffix = "." + suffix
        return figure.id.replace(":", "_") + suffix

    def figure_paths_map(self, questions: Sequence[EnrichedQuestion]) -> dict[str, str]:
        """Maps ``fig:id`` → local asset name for the transpiler."""
        mapping: dict[str, str] = {}
        for question in questions:
            for figure in question.figures:
                mapping[figure.id] = self.local_asset_name(figure)
        return mapping

    # -- Localization -------------------------------------------------------
    def localized_statement(self, question: EnrichedQuestion, locale: str | None) -> str:
        """Statement in ``locale`` if a validated translated copy exists; otherwise the original.

        The canonical statement lives in ``questions`` and is the only one that
        takes part in deduplication; here only the presentation variant is
        chosen.
        """
        if locale:
            translation = self.database.get_question_translation(question.id, locale)
            if translation is not None:
                return translation.statement_raw
        return question.statement_raw

    # -- Templates ----------------------------------------------------------
    def build_exam_typst(
        self,
        questions: Sequence[EnrichedQuestion],
        *,
        title: str = "Examen de práctica",
        figure_paths: Mapping[str, str] | None = None,
        locale: str | None = None,
    ) -> str:
        """Assembles the Typst source of the exam from the statements."""
        paths = dict(figure_paths or self.figure_paths_map(questions))
        lines = [f"= {title}", ""]
        for index, question in enumerate(questions, start=1):
            lines.append(f"== Pregunta {index}")
            statement = self.localized_statement(question, locale)
            lines.append(transpiler.markdown_to_typst(statement, figure_paths=paths).rstrip())
            lines.append("")
        return "\n".join(lines) + "\n"

    def build_exam_latex(
        self,
        questions: Sequence[EnrichedQuestion],
        *,
        title: str = "Examen de práctica",
        figure_paths: Mapping[str, str] | None = None,
        locale: str | None = None,
    ) -> str:
        """Assembles the LaTeX source of the exam using ``\\includegraphics``."""
        paths = dict(figure_paths or self.figure_paths_map(questions))
        lines = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{graphicx}",
            "\\usepackage{amsmath,amssymb}",
            "\\usepackage[spanish]{babel}",
            "\\begin{document}",
            f"\\title{{{title}}}",
            "\\maketitle",
        ]
        for index, question in enumerate(questions, start=1):
            lines.append(f"\\section*{{Pregunta {index}}}")
            statement = self.localized_statement(question, locale)
            lines.append(transpiler.markdown_to_latex(statement, figure_paths=paths).rstrip())
        lines.append("\\end{document}")
        return "\n".join(lines) + "\n"

    # -- Assets -------------------------------------------------------------
    def copy_assets(
        self, questions: Sequence[EnrichedQuestion], assets_dir: str | Path
    ) -> dict[str, str]:
        """Downloads the used figures from SeaweedFS to ``assets_dir``.

        Only copies the figures referenced by the statements; it does not keep a
        second permanent local copy.
        """
        directory = Path(assets_dir)
        directory.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, str] = {}
        for question in questions:
            for figure in question.figures:
                if not figure.asset_path:
                    continue
                local_name = self.local_asset_name(figure)
                mapping[figure.id] = local_name
                destination = directory / local_name
                if destination.exists():
                    continue
                bucket, key = split_asset_reference(
                    figure.asset_path, default_bucket=self.settings.S3_BUCKET
                )
                self.storage.download_to_file(key, destination, bucket=bucket)
        return mapping

    # -- Generation ---------------------------------------------------------
    def generate_exam(
        self,
        *,
        course: str | None = None,
        topics: Sequence[str] | None = None,
        difficulty: int | None = None,
        num_questions: int = 10,
        exam_id: str | None = None,
        title: str | None = None,
        downloads_dir: str | Path | None = None,
        locale: str | None = None,
    ) -> ExamResult:
        """Generates and compiles a complete practice exam.

        ``locale`` selects, when available, the translated copy of the statement
        and the localized course name; if missing, the canonical statement is
        used. It reports explicitly when not enough questions were retrieved.
        """
        downloads = Path(downloads_dir or self.settings.DOWNLOADS_DIR)
        exams_dir = downloads / "exams"
        exams_dir.mkdir(parents=True, exist_ok=True)

        questions = self.retrieve_questions(
            course=course, topics=topics, difficulty=difficulty, limit=num_questions
        )
        warnings: list[str] = []
        status = "ok"
        if len(questions) < num_questions:
            status = "partial"
            warnings.append(
                f"only {len(questions)} of the {num_questions} requested questions were found"
            )
        if not questions:
            return ExamResult(exam_id or "", "error", 0, None, None, warnings)

        identifier = exam_id or uuid.uuid4().hex
        if not _is_valid_exam_id(identifier):
            raise ExamGenerationError("invalid exam_id")

        source_dir = exams_dir / identifier
        assets_dir = source_dir / "assets"
        source_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        figure_paths = self.copy_assets(questions, assets_dir)
        active_locale = locale or DEFAULT_LOCALE
        if title:
            resolved_title = title
        elif course:
            course_name = self.database.get_course_name(
                course, locale=active_locale
            ) or format_course_name(course, locale=active_locale)
            resolved_title = f"Examen de práctica — {course_name}"
        else:
            resolved_title = "Examen de práctica"
        try:
            typst = self.build_exam_typst(
                questions, title=resolved_title, figure_paths=figure_paths, locale=locale
            )
            (source_dir / "main.typ").write_text(typst)
            manifest = {
                "exam_id": identifier,
                "title": resolved_title,
                "figure_paths": figure_paths,
                "questions": [question.model_dump(mode="json") for question in questions],
            }
            (source_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
            compile_result = transpiler.compile_typst(typst, source_dir, output_name="output.pdf")
        except transpiler.TranspileError as exc:
            logger.error(
                "Markdown/TeX→Typst conversion error for exam %s: %s%s",
                identifier,
                exc.message,
                f" (original formula: {exc.original!r})" if exc.original else "",
            )
            return ExamResult(
                identifier,
                "error",
                len(questions),
                None,
                None,
                [f"Typst conversion error: {exc.message}"],
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error(
                "could not run pandoc/typst while generating exam %s: %s",
                identifier,
                exc,
            )
            return ExamResult(
                identifier,
                "error",
                len(questions),
                None,
                None,
                [f"Typst compilation error: {exc}"],
            )
        if not compile_result.ok:
            logger.error(
                "Typst rejected the source of exam %s:\n%s",
                identifier,
                compile_result.format_errors(),
            )
            return ExamResult(
                identifier,
                "error",
                len(questions),
                None,
                None,
                [compile_result.format_errors()],
            )

        pdf_path = exams_dir / f"{identifier}.pdf"
        if compile_result.pdf_path is not None:
            shutil.move(str(compile_result.pdf_path), str(pdf_path))
        pdf_url = f"{self.settings.DOWNLOAD_PUBLIC_URL.rstrip('/')}/download/exams/{identifier}.pdf"
        return ExamResult(identifier, status, len(questions), pdf_path, pdf_url, warnings)

    # -- Source packaging ---------------------------------------------------
    def build_source_zip(
        self,
        exam_id: str,
        *,
        format: str = "typst",
        downloads_dir: str | Path | None = None,
    ) -> Path:
        """Generates the ZIP with the exam source (``main.typ``/``main.tex`` + assets).

        Validates ``exam_id`` and ``format`` to prevent traversal or arbitrary reads.
        """
        if format not in _ALLOWED_FORMATS:
            raise ExamGenerationError(f"unsupported format: {format!r}")
        if not _is_valid_exam_id(exam_id):
            raise ExamGenerationError("invalid exam_id")

        downloads = Path(downloads_dir or self.settings.DOWNLOADS_DIR)
        source_dir = downloads / "exams" / exam_id
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ExamGenerationError(f"no exam exists with id {exam_id!r}")

        manifest = json.loads(manifest_path.read_text())
        zip_path = downloads / f"fuente_examen_{exam_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            if format == "latex":
                archive.writestr("main.tex", self._manifest_to_latex(manifest))
            else:
                archive.write(source_dir / "main.typ", "main.typ")
            for asset in sorted((source_dir / "assets").glob("*")):
                if asset.is_file():
                    archive.write(asset, f"assets/{asset.name}")
            archive.writestr("README.txt", self._readme(format))
        return zip_path

    def _manifest_to_latex(self, manifest: dict[str, Any]) -> str:
        questions = [EnrichedQuestion.model_validate(q) for q in manifest["questions"]]
        return self.build_exam_latex(
            questions,
            title=manifest.get("title", "Examen de práctica"),
            figure_paths=manifest.get("figure_paths", {}),
        )

    @staticmethod
    def _readme(format: str) -> str:
        if format == "latex":
            return (
                "Fuente del examen en LaTeX.\n\n"
                "Compilar con: pdflatex main.tex\n\n"
                "Las figuras utilizadas están en la carpeta assets/.\n"
            )
        return (
            "Fuente del examen en Typst.\n\n"
            "Compilar con: typst compile main.typ output.pdf\n\n"
            "Las figuras utilizadas están en la carpeta assets/.\n"
        )
