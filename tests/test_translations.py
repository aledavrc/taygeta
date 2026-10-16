"""Tests for localized question copies (storage, validation and exam selection)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taygeta.agents.translator import (
    TranslationBundle,
    TranslationDraft,
    Translator,
)
from taygeta.core.schema import Choice, EnrichedQuestion, FigureReference, utc_now_iso
from taygeta.domain.translation import QuestionTranslation
from taygeta.ingest.validator import validate_translation
from taygeta.server.db import Database
from taygeta.server.exam_generator import ExamGenerator
from taygeta.server.translation_service import TranslationService


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(db_path=tmp_path / "translations.db")


def _question(**overrides: Any) -> EnrichedQuestion:
    payload: dict[str, Any] = {
        "id": str(uuid4()),
        "statement_raw": "Calcula el valor de $x^2 + 1$ cuando $x = 3$.",
        "question_type": "open_ended",
        "choices": None,
        "figures": [],
        "notation_notes": [],
        "course": "algebra",
        "topic": "expresiones",
        "subtopic": None,
        "difficulty": 1,
        "prerequisites": [],
        "source_metadata": {"source_reference": "doc.pdf"},
        "created_at": utc_now_iso(),
    }
    payload.update(overrides)
    return EnrichedQuestion(**payload)


def _translation(question: EnrichedQuestion, **overrides: Any) -> QuestionTranslation:
    payload: dict[str, Any] = {
        "question_id": question.id,
        "locale": "en",
        "statement_raw": "Compute the value of $x^2 + 1$ when $x = 3$.",
        "choices": None,
        "created_at": utc_now_iso(),
    }
    payload.update(overrides)
    return QuestionTranslation(**payload)


# -- Model ------------------------------------------------------------------


def test_translation_rejects_unsupported_locale() -> None:
    question = _question()
    with pytest.raises(ValidationError):
        _translation(question, locale="fr")


# -- Validation -------------------------------------------------------------


def test_validate_translation_accepts_faithful_copy() -> None:
    question = _question()
    result = validate_translation(question, _translation(question))
    assert result.is_valid
    assert result.checks["math_preserved"]


def test_validate_translation_rejects_altered_math() -> None:
    question = _question()
    broken = _translation(question, statement_raw="Compute the value of $x^3 + 1$ when $x = 3$.")
    result = validate_translation(question, broken)
    assert not result.is_valid
    assert any(issue.code == "math_not_preserved" for issue in result.issues)


def test_validate_translation_rejects_unbalanced_delimiters() -> None:
    question = _question()
    broken = _translation(question, statement_raw="Compute the value of $x^2 + 1$ when x = 3.")
    result = validate_translation(question, broken)
    assert not result.is_valid


def test_validate_translation_checks_figure_placeholders() -> None:
    figure = FigureReference(id="fig:1", page=0, bbox=[0, 0, 100, 100])
    question = _question(statement_raw="Observa {{fig:1}} y calcula $x$.", figures=[figure])
    # Missing placeholder in the translation.
    missing = _translation(question, statement_raw="Look at the figure and compute $x$.")
    assert not validate_translation(question, missing).is_valid
    # Placeholder preserved.
    kept = _translation(question, statement_raw="Look at {{fig:1}} and compute $x$.")
    assert validate_translation(question, kept).is_valid


def test_validate_translation_requires_same_choice_ids() -> None:
    question = _question(
        question_type="multiple_choice",
        choices=[Choice(id="A", content_raw="Uno"), Choice(id="B", content_raw="Dos")],
    )
    broken = _translation(question, choices=[Choice(id="A", content_raw="One")])
    result = validate_translation(question, broken)
    assert not result.is_valid
    assert any(issue.code == "translation_choices_mismatch" for issue in result.issues)
    faithful = _translation(
        question, choices=[Choice(id="A", content_raw="One"), Choice(id="B", content_raw="Two")]
    )
    assert validate_translation(question, faithful).is_valid


# -- Translator assembly (no network) ---------------------------------------


def test_assemble_builds_one_copy_per_locale() -> None:
    question = _question()
    bundle = TranslationBundle(
        es=TranslationDraft(statement_raw="Calcula $x^2 + 1$."),
        en=TranslationDraft(statement_raw="Compute $x^2 + 1$."),
    )
    translations = Translator._assemble(question, bundle)
    assert {entry.locale for entry in translations} == {"es", "en"}
    assert all(entry.question_id == question.id for entry in translations)


# -- Persistence + FTS ------------------------------------------------------


def test_upsert_and_get_translation(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    database.upsert_question_translation(_translation(question))
    stored = database.get_question_translation(question.id, "en")
    assert stored is not None and stored.statement_raw.startswith("Compute")
    assert database.get_question_translation(question.id, "es") is None


def test_search_translation_ids_filters_by_locale(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    database.upsert_question_translation(_translation(question))
    assert database.search_translation_ids("Compute", locale="en") == [question.id]
    assert database.search_translation_ids("Compute", locale="es") == []


def test_delete_question_removes_translations_and_fts(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    database.upsert_question_translation(_translation(question))
    assert database.delete_question(question.id) is True
    assert database.get_question_translation(question.id, "en") is None
    assert database.search_translation_ids("Compute", locale="en") == []


# -- Service ----------------------------------------------------------------


class _FakeTranslator:
    def __init__(self, translations: list[QuestionTranslation]) -> None:
        self._translations = translations

    def translate(self, question: EnrichedQuestion) -> list[QuestionTranslation]:
        return [
            entry.model_copy(update={"question_id": question.id}) for entry in self._translations
        ]


def test_service_stores_only_valid_copies(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    valid = _translation(question, locale="en")
    invalid = _translation(question, locale="es", statement_raw="Calcula $x^3 + 1$.")
    service = TranslationService(database, _FakeTranslator([valid, invalid]))
    outcome = service.store(question)
    assert outcome.stored == ["en"]
    assert outcome.problems and "es" in outcome.problems[0]
    assert database.get_question_translation(question.id, "en") is not None


def test_service_reports_provider_failure(database: Database) -> None:
    question = _question()

    class _Broken:
        def translate(self, question: EnrichedQuestion) -> list[QuestionTranslation]:
            raise RuntimeError("sin proveedor")

    outcome = TranslationService(database, _Broken()).store(question)
    assert outcome.stored == []
    assert outcome.problems == [f"{question.id}: could not translate (sin proveedor)"]


def test_service_rejects_copies_that_fail_semantic_validation(database: Database) -> None:
    question = _question()
    database.upsert_question(question)

    class _RejectingSemantic:
        def validate(self, candidate: Any) -> Any:
            class _Result:
                is_valid = False

            return _Result()

    service = TranslationService(
        database,
        _FakeTranslator([_translation(question, locale="en")]),
        semantic=_RejectingSemantic(),
    )
    outcome = service.store(question)
    assert outcome.stored == []
    assert "semantic validation" in outcome.problems[0]
    assert database.get_question_translation(question.id, "en") is None


# -- Exam locale selection --------------------------------------------------


def test_localized_statement_prefers_translation(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    database.upsert_question_translation(_translation(question))
    generator = ExamGenerator(database=database)
    assert generator.localized_statement(question, "en").startswith("Compute")
    assert generator.localized_statement(question, "es") == question.statement_raw
    assert generator.localized_statement(question, None) == question.statement_raw
