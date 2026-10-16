"""Tests for the localized course-name presentation helpers (``domain.taxonomy``)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from taygeta.domain.taxonomy import format_course_name, title_case_slug
from taygeta.i18n import set_locale_resolver, set_taxonomy_name_provider


@pytest.fixture(autouse=True)
def _reset_globals() -> Iterator[None]:
    """Ensure every test starts and ends without a resolver or name provider."""
    set_locale_resolver(None)
    set_taxonomy_name_provider(None)
    yield
    set_locale_resolver(None)
    set_taxonomy_name_provider(None)


def _provider(names: dict[tuple[str, str, str, str], str]):
    def provider(kind: str, slug: str, locale: str, parent_slug: str = "") -> str | None:
        return names.get((kind, parent_slug, slug, locale))

    return provider


def test_format_course_name_uses_active_locale() -> None:
    set_taxonomy_name_provider(
        _provider({("course", "", "calculo_diferencial", "en"): "Differential Calculus"})
    )
    set_locale_resolver(lambda: "en")
    assert format_course_name("calculo_diferencial") == "Differential Calculus"


def test_format_course_name_accepts_explicit_locale() -> None:
    set_taxonomy_name_provider(
        _provider({("course", "", "calculo_diferencial", "es"): "Cálculo Diferencial"})
    )
    assert format_course_name("calculo_diferencial", locale="es") == "Cálculo Diferencial"


def test_format_course_name_falls_back_to_title_case() -> None:
    assert format_course_name("metodos_numericos", locale="en") == "Metodos Numericos"


def test_title_case_slug_is_reexported() -> None:
    assert title_case_slug("algebra_lineal", locale="en") == "Algebra Lineal"
