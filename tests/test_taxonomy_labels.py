"""Tests for DB-backed taxonomy labels (schema v3, ``taxonomy_labels``)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from taygeta.domain.taxonomy import format_course_name
from taygeta.i18n import set_taxonomy_name_provider
from taygeta.server.db import Database
from taygeta.ui.nice.taxonomy import install_taxonomy_names, invalidate_taxonomy_names


@pytest.fixture(autouse=True)
def _reset_provider() -> Iterator[None]:
    set_taxonomy_name_provider(None)
    yield
    set_taxonomy_name_provider(None)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(db_path=tmp_path / "taxonomy.db")


def test_seed_populates_course_labels(database: Database) -> None:
    assert database.get_course_name("algebra", locale="es") == "Álgebra"
    assert database.get_course_name("algebra", locale="en") == "Algebra"


def test_seed_is_idempotent(database: Database) -> None:
    database.initialize()  # a second run must neither duplicate nor raise
    labels = database.taxonomy_label_map(locale="es")
    assert labels[("course", "", "algebra")] == "Álgebra"


def test_upsert_label_round_trip(database: Database) -> None:
    database.upsert_course("metodos_numericos")
    database.upsert_taxonomy_label(
        "course", "metodos_numericos", "Métodos Numéricos", locale="es", source="manual"
    )
    assert (
        database.get_taxonomy_label("course", "metodos_numericos", locale="es")
        == "Métodos Numéricos"
    )


def test_label_source_precedence(database: Database) -> None:
    # An LLM label never overwrites the seed.
    database.upsert_taxonomy_label("course", "algebra", "LLM NAME", locale="en", source="llm")
    assert database.get_course_name("algebra", locale="en") == "Algebra"
    # A manual label wins over everything.
    database.upsert_taxonomy_label("course", "algebra", "Manual", locale="en", source="manual")
    assert database.get_course_name("algebra", locale="en") == "Manual"


def test_topic_labels_use_parent_context(database: Database) -> None:
    database.upsert_course("algebra")
    database.upsert_taxonomy_label(
        "topic", "ecuaciones", "Ecuaciones", locale="es", parent_slug="algebra"
    )
    assert (
        database.get_taxonomy_label("topic", "ecuaciones", parent_slug="algebra", locale="es")
        == "Ecuaciones"
    )


def test_list_taxonomy_is_localized(database: Database) -> None:
    taxonomy = database.list_taxonomy()
    algebra = next(course for course in taxonomy["courses"] if course["slug"] == "algebra")
    assert algebra["names"] == {"es": "Álgebra", "en": "Algebra"}


def test_provider_install_and_fallback(database: Database) -> None:
    install_taxonomy_names(database)
    assert format_course_name("algebra", locale="en") == "Algebra"
    assert format_course_name("desconocido", locale="en") == "Desconocido"


def test_provider_cache_invalidates(database: Database) -> None:
    install_taxonomy_names(database)
    assert format_course_name("algebra", locale="en") == "Algebra"
    database.upsert_taxonomy_label("course", "algebra", "Manual", locale="en", source="manual")
    # The provider still serves the cached map until it is invalidated.
    assert format_course_name("algebra", locale="en") == "Algebra"
    invalidate_taxonomy_names()
    assert format_course_name("algebra", locale="en") == "Manual"
