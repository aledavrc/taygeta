"""Tests for LLM-generated taxonomy names carried from structuring to confirmation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from taygeta.agents.structurer import (
    _REFINING_SYSTEM_PROMPT,
    _STRUCTURING_SYSTEM_PROMPT,
    QuestionDraft,
    Structurer,
    _build_dynamic_curriculum_prompt,
)
from taygeta.domain.taxonomy import TaxonomyNames, filter_taxonomy_names
from taygeta.server.db import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(db_path=tmp_path / "taxonomy_names.db")


def _names(
    kind: str, slug: str, *, parent_slug: str = "", es: str = "Es", en: str = "En"
) -> TaxonomyNames:
    return TaxonomyNames(kind=kind, slug=slug, parent_slug=parent_slug, es=es, en=en)  # type: ignore[arg-type]


def test_locale_labels_pairs() -> None:
    entry = _names("course", "algebra", es="Álgebra", en="Algebra")
    assert entry.locale_labels() == [("es", "Álgebra"), ("en", "Algebra")]


def test_filter_keeps_matching_slugs_and_reparents() -> None:
    names = [
        _names("course", "algebra"),
        _names("topic", "matrices", parent_slug="algebra"),
        _names("subtopic", "determinantes", parent_slug="algebra"),
        _names("prerequisite", "sumas"),
        _names("course", "calculo"),
    ]
    kept = filter_taxonomy_names(
        names,
        course="algebra",
        topic="matrices",
        subtopic="determinantes",
        prerequisites=["sumas"],
    )
    assert {(entry.kind, entry.slug) for entry in kept} == {
        ("course", "algebra"),
        ("topic", "matrices"),
        ("subtopic", "determinantes"),
        ("prerequisite", "sumas"),
    }
    topic = next(entry for entry in kept if entry.kind == "topic")
    assert topic.parent_slug == "algebra"


def test_filter_drops_edited_or_removed_entries() -> None:
    names = [
        _names("course", "algebra"),
        _names("topic", "matrices"),
        _names("subtopic", "determinantes"),
        _names("prerequisite", "sumas"),
    ]
    # The reviewer changed the topic, removed the subtopic and the prerequisite.
    kept = filter_taxonomy_names(
        names, course="algebra", topic="espacios_vectoriales", subtopic=None, prerequisites=[]
    )
    assert [entry.kind for entry in kept] == ["course"]


def _draft(**overrides: Any) -> QuestionDraft:
    payload: dict[str, Any] = {
        "statement_raw": "Calcula el determinante de la matriz.",
        "question_type": "open_ended",
        "course": "algebra_lineal",
        "topic": "matrices",
        "subtopic": "determinantes",
        "difficulty": 2,
        "prerequisites": ["sumas", "productos"],
        "course_name": {"es": "Álgebra Lineal", "en": "Linear Algebra"},
        "topic_name": {"es": "Matrices", "en": "Matrices"},
        "subtopic_name": {"es": "Determinantes", "en": "Determinants"},
        "prerequisite_names": [
            {"slug": "sumas", "es": "Sumas", "en": "Sums"},
            {"slug": "no_existe", "es": "Fantasma", "en": "Ghost"},
        ],
    }
    payload.update(overrides)
    return QuestionDraft(**payload)


def test_collect_taxonomy_names_projects_roles() -> None:
    entries = Structurer._collect_taxonomy_names(_draft())
    by_key = {(entry.kind, entry.slug): entry for entry in entries}
    assert set(by_key) == {
        ("course", "algebra_lineal"),
        ("topic", "matrices"),
        ("subtopic", "determinantes"),
        ("prerequisite", "sumas"),
    }
    assert by_key[("course", "algebra_lineal")].parent_slug == ""
    assert by_key[("topic", "matrices")].parent_slug == "algebra_lineal"
    assert by_key[("subtopic", "determinantes")].parent_slug == "algebra_lineal"
    assert by_key[("prerequisite", "sumas")].en == "Sums"
    # A prerequisite name whose slug is not in the list is dropped.
    assert ("prerequisite", "no_existe") not in by_key


def test_collect_taxonomy_names_ignores_subtopic_name_without_subtopic() -> None:
    entries = Structurer._collect_taxonomy_names(
        _draft(subtopic=None, subtopic_name={"es": "Algo", "en": "Something"})
    )
    assert all(entry.kind != "subtopic" for entry in entries)


def test_collect_taxonomy_names_skips_blank_names() -> None:
    entries = Structurer._collect_taxonomy_names(_draft(course_name={"es": "  ", "en": "   "}))
    assert all(entry.kind != "course" for entry in entries)


def test_upsert_taxonomy_names_never_clobbers_seed(database: Database) -> None:
    # Seeded names (source='seed') win over generated ones.
    database.upsert_taxonomy_names(
        [_names("course", "algebra", es="Álgebra LLM", en="Algebra LLM")]
    )
    assert database.get_course_name("algebra", locale="es") == "Álgebra"


def test_upsert_taxonomy_names_stores_new_slugs(database: Database) -> None:
    database.upsert_course("metodos_numericos")
    written = database.upsert_taxonomy_names(
        [
            _names(
                "course",
                "metodos_numericos",
                es="Métodos Numéricos",
                en="Numerical Methods",
            )
        ]
    )
    assert written == 2
    assert database.get_course_name("metodos_numericos", locale="en") == "Numerical Methods"
    assert (
        database.get_taxonomy_label("course", "metodos_numericos", locale="es")
        == "Métodos Numéricos"
    )


def test_prompts_request_localized_names() -> None:
    for prompt in (_STRUCTURING_SYSTEM_PROMPT, _REFINING_SYSTEM_PROMPT):
        assert "course_name" in prompt
        assert "prerequisite_names" in prompt


def test_dynamic_curriculum_prompt_annotates_existing_names() -> None:
    tree = {"algebra": {"matrices": ["determinantes"]}}
    names = {
        ("course", "", "algebra"): {"es": "Álgebra", "en": "Algebra"},
        ("topic", "algebra", "matrices"): {"es": "Matrices", "en": "Matrices"},
    }
    text = _build_dynamic_curriculum_prompt(tree, names)
    assert 'es: "Álgebra"' in text
    assert 'en: "Matrices"' in text


def test_staging_data_serializes_taxonomy_names() -> None:
    from taygeta.core.schema import EnrichedQuestion, utc_now_iso
    from taygeta.ingest.pipeline import IngestPipeline, StagedQuestion, StagingResult

    question = EnrichedQuestion(
        id=str(uuid4()),
        statement_raw="Calcula $x$.",
        question_type="open_ended",
        course="algebra",
        topic="matrices",
        subtopic=None,
        difficulty=2,
        prerequisites=[],
        created_at=utc_now_iso(),
    )
    result = StagingResult(
        source_reference="doc.pdf",
        status="VALID",
        questions=[
            StagedQuestion(
                question=question,
                status="VALID",
                taxonomy_names=[_names("course", "algebra", es="Álgebra", en="Algebra")],
            )
        ],
    )
    data = IngestPipeline._staging_data(None, result)  # type: ignore[arg-type]
    assert data["questions"][0]["taxonomy_names"][0]["en"] == "Algebra"
