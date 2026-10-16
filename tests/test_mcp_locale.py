"""Tests for the MCP language plumbing (localized taxonomy and question payloads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from taygeta.core.schema import EnrichedQuestion, utc_now_iso
from taygeta.core.settings import get_settings
from taygeta.domain.translation import QuestionTranslation
from taygeta.server.db import Database
from taygeta.server.mcp_server import _public_question


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(db_path=tmp_path / "mcp_locale.db")


def _question(**overrides: Any) -> EnrichedQuestion:
    payload: dict[str, Any] = {
        "id": str(uuid4()),
        "statement_raw": "Calcula $x^2 + 1$ cuando $x = 3$.",
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


def _translation(question: EnrichedQuestion, statement: str) -> QuestionTranslation:
    return QuestionTranslation(
        question_id=question.id,
        locale="en",
        statement_raw=statement,
        created_at=utc_now_iso(),
    )


# -- list_taxonomy ----------------------------------------------------------


def test_list_taxonomy_default_returns_names_maps(database: Database) -> None:
    data = database.list_taxonomy()
    assert "locale" not in data
    courses = {course["slug"]: course for course in data["courses"]}
    assert courses["algebra"]["names"] == {"es": "Álgebra", "en": "Algebra"}


def test_list_taxonomy_localized_returns_flat_names(database: Database) -> None:
    database.upsert_topic("algebra", "matrices", "determinantes")
    data = database.list_taxonomy(locale="en")
    assert data["locale"] == "en"
    courses = {course["slug"]: course for course in data["courses"]}
    assert courses["algebra"] == {"slug": "algebra", "name": "Algebra"}
    topics = {(topic["course_slug"], topic["topic_slug"]): topic for topic in data["topics"]}
    assert topics[("algebra", "matrices")]["name"] == "Matrices"
    assert topics[("algebra", "matrices")]["subtopic_name"] == "Determinantes"


def test_list_taxonomy_ignores_unsupported_locale(database: Database) -> None:
    data = database.list_taxonomy(locale="fr")
    assert "locale" not in data
    assert "names" in data["courses"][0]


# -- _public_question -------------------------------------------------------


def test_public_question_returns_all_locales_by_default(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    database.upsert_question_translation(_translation(question, "Compute $x^2 + 1$."))
    data = _public_question(question, get_settings(), database)
    assert data["statement_by_locale"] == {"en": "Compute $x^2 + 1$."}
    assert "statement" not in data
    assert data["statement_raw"] == question.statement_raw


def test_public_question_resolves_requested_locale(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    database.upsert_question_translation(_translation(question, "Compute $x^2 + 1$."))
    data = _public_question(question, get_settings(), database, locale="en")
    assert data["locale"] == "en"
    assert data["statement"].startswith("Compute")


def test_public_question_falls_back_to_canonical_statement(database: Database) -> None:
    question = _question()
    database.upsert_question(question)
    data = _public_question(question, get_settings(), database, locale="en")
    assert data["locale"] == "en"
    assert data["statement"] == question.statement_raw


def test_public_question_without_database_has_no_locales() -> None:
    question = _question()
    data = _public_question(question, get_settings())
    assert "statement_by_locale" not in data
    assert "statement" not in data
