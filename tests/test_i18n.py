"""Tests for the framework-agnostic i18n layer (``taygeta.i18n``)."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest

import taygeta.i18n as i18n
from taygeta.i18n import (
    DEFAULT_LOCALE,
    LOCALE_LABELS,
    SUPPORTED_LOCALES,
    course_name,
    get_locale,
    is_supported_locale,
    set_locale_resolver,
    set_taxonomy_name_provider,
    t,
    taxonomy_name,
    title_case_slug,
)
from taygeta.i18n.catalog_en import CATALOG as CATALOG_EN
from taygeta.i18n.catalog_es import CATALOG as CATALOG_ES


@pytest.fixture(autouse=True)
def _reset_globals() -> Iterator[None]:
    """Ensure every test starts and ends without a resolver or name provider."""
    set_locale_resolver(None)
    set_taxonomy_name_provider(None)
    yield
    set_locale_resolver(None)
    set_taxonomy_name_provider(None)


def test_catalogs_share_the_same_keys() -> None:
    assert set(CATALOG_ES) == set(CATALOG_EN)


def test_catalogs_use_the_same_placeholders() -> None:
    pattern = re.compile(r"\{(\w+)\}")
    for key, text in CATALOG_ES.items():
        assert set(pattern.findall(text)) == set(pattern.findall(CATALOG_EN[key])), key


def test_default_locale_is_supported() -> None:
    assert DEFAULT_LOCALE in SUPPORTED_LOCALES


def test_locale_labels_cover_every_supported_locale() -> None:
    assert set(LOCALE_LABELS) == set(SUPPORTED_LOCALES)


def test_is_supported_locale() -> None:
    assert is_supported_locale("en")
    assert not is_supported_locale("fr")
    assert not is_supported_locale(None)


def test_get_locale_defaults_without_resolver() -> None:
    assert get_locale() == DEFAULT_LOCALE


def test_resolver_overrides_default() -> None:
    set_locale_resolver(lambda: "en")
    assert get_locale() == "en"


def test_unsupported_resolver_value_falls_back() -> None:
    set_locale_resolver(lambda: "fr")
    assert get_locale() == DEFAULT_LOCALE


def test_resolver_error_falls_back() -> None:
    def boom() -> str:
        raise RuntimeError("no client context")

    set_locale_resolver(boom)
    assert get_locale() == DEFAULT_LOCALE


def test_translate_uses_explicit_locale() -> None:
    key = next(iter(CATALOG_EN))
    assert t(key, locale="en") == CATALOG_EN[key]


def test_translate_uses_active_locale_from_resolver() -> None:
    key = next(iter(CATALOG_EN))
    set_locale_resolver(lambda: "en")
    assert t(key) == CATALOG_EN[key]


def test_translate_unknown_locale_returns_key() -> None:
    assert t("does.not.exist", locale="en") == "does.not.exist"


def test_translate_falls_back_to_default_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    key = next(iter(CATALOG_ES))
    monkeypatch.delitem(i18n._CATALOGS["en"], key)
    assert t(key, locale="en") == CATALOG_ES[key]


def test_translate_formats_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(i18n._CATALOGS["en"], "greeting", "Hello, {name}!")
    assert t("greeting", locale="en", name="Ada") == "Hello, Ada!"


def test_translate_keeps_unknown_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(i18n._CATALOGS["en"], "raw", "value: {missing}")
    assert t("raw", locale="en") == "value: {missing}"


def test_course_name_uses_provider() -> None:
    def provider(kind: str, slug: str, locale: str, parent_slug: str = "") -> str | None:
        names = {
            ("course", "", "algebra_lineal", "es"): "Álgebra Lineal",
            ("course", "", "algebra_lineal", "en"): "Linear Algebra",
        }
        return names.get((kind, parent_slug, slug, locale))

    set_taxonomy_name_provider(provider)
    assert course_name("algebra_lineal", locale="es") == "Álgebra Lineal"
    assert course_name("algebra_lineal", locale="en") == "Linear Algebra"


def test_course_name_provider_falls_back_to_default_locale() -> None:
    def provider(kind: str, slug: str, locale: str, parent_slug: str = "") -> str | None:
        if (kind, slug, locale) == ("course", "algebra", "es"):
            return "Álgebra"
        return None

    set_taxonomy_name_provider(provider)
    assert course_name("algebra", locale="en") == "Álgebra"


def test_taxonomy_name_returns_none_without_provider() -> None:
    assert taxonomy_name("course", "algebra", locale="es") is None


def test_course_name_falls_back_to_title_case_in_the_same_locale() -> None:
    assert course_name("metodos_numericos", locale="en") == "Metodos Numericos"


def test_course_name_uses_active_locale() -> None:
    def provider(kind: str, slug: str, locale: str, parent_slug: str = "") -> str | None:
        if (kind, slug, locale) == ("course", "topologia", "en"):
            return "Topology"
        return None

    set_taxonomy_name_provider(provider)
    set_locale_resolver(lambda: "en")
    assert course_name("topologia") == "Topology"


def test_course_name_survives_provider_errors() -> None:
    def provider(kind: str, slug: str, locale: str, parent_slug: str = "") -> str | None:
        raise RuntimeError("no database")

    set_taxonomy_name_provider(provider)
    assert course_name("algebra", locale="en") == "Algebra"


def test_title_case_slug_uses_locale_minor_words() -> None:
    assert title_case_slug("analisis_de_ecuaciones", locale="es") == "Analisis de Ecuaciones"
    assert title_case_slug("theory_of_groups", locale="en") == "Theory of Groups"


def test_title_case_slug_capitalizes_a_leading_minor_word() -> None:
    assert title_case_slug("de_la_cruz", locale="es") == "De la Cruz"
