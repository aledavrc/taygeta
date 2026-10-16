"""Internationalization (i18n) for the Taygeta interface.

Framework-agnostic translation layer for every user-facing string. Catalogs
live in :mod:`taygeta.i18n.catalog_es` and :mod:`taygeta.i18n.catalog_en` and
are keyed by dotted names (``area.subject``).

The active locale is resolved through a pluggable callable (see
:func:`set_locale_resolver`) so the UI can read it from the session without
this module depending on NiceGUI. When no resolver is installed, or the stored
value is unsupported, :data:`DEFAULT_LOCALE` is used.

Lookups fall back gracefully: active locale → default locale → the key itself,
so a missing translation never breaks rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from taygeta.i18n.catalog_en import CATALOG as _CATALOG_EN
from taygeta.i18n.catalog_es import CATALOG as _CATALOG_ES

__all__ = [
    "DEFAULT_LOCALE",
    "LOCALE_LABELS",
    "SUPPORTED_LOCALES",
    "course_name",
    "get_locale",
    "is_supported_locale",
    "set_locale_resolver",
    "set_taxonomy_name_provider",
    "t",
    "taxonomy_name",
    "title_case_slug",
]

#: Locales the interface can be rendered in.
SUPPORTED_LOCALES: tuple[str, ...] = ("es", "en")

#: Locale used when nothing else is configured or the stored value is invalid.
DEFAULT_LOCALE: str = "es"

#: Self-described names for each locale, shown in the language chooser. Each
#: name is written in its own language, independent of the active locale.
LOCALE_LABELS: dict[str, str] = {
    "es": "Español",
    "en": "English",
}

_CATALOGS: dict[str, dict[str, str]] = {
    "es": _CATALOG_ES,
    "en": _CATALOG_EN,
}

#: Optional callable that resolves a taxonomy display name from data (typically a
#: database-backed adapter installed by the UI). Called as
#: ``provider(kind, slug, locale, parent_slug) `` and may return ``None`` or raise
#: when no name is known; in that case resolution falls back to ``title_case_slug``.
_taxonomy_name_provider: Callable[..., str | None] | None = None

#: Minor words kept lowercase (except when they head a title) when turning a
#: ``snake_case`` slug into a human-readable name, per locale.
_TITLE_CASE_MINOR: dict[str, frozenset[str]] = {
    "es": frozenset(
        {
            "de",
            "del",
            "el",
            "la",
            "las",
            "los",
            "y",
            "e",
            "o",
            "u",
            "a",
            "en",
            "con",
            "para",
            "por",
            "sin",
            "sobre",
            "entre",
        }
    ),
    "en": frozenset(
        {
            "of",
            "the",
            "and",
            "or",
            "a",
            "an",
            "in",
            "on",
            "with",
            "for",
            "to",
            "by",
            "at",
            "from",
            "as",
            "into",
            "over",
            "under",
            "between",
            "near",
            "but",
            "nor",
            "per",
            "via",
        }
    ),
}

_locale_resolver: Callable[[], str | None] | None = None


def is_supported_locale(locale: str | None) -> bool:
    """Return whether ``locale`` is one of :data:`SUPPORTED_LOCALES`."""
    return locale in SUPPORTED_LOCALES


def set_locale_resolver(resolver: Callable[[], str | None] | None) -> None:
    """Install (or clear) the callable that reports the active locale.

    The resolver typically reads the locale stored for the current session. It
    may raise or return ``None`` when no session exists; in that case the
    default locale is used.
    """
    global _locale_resolver
    _locale_resolver = resolver


def get_locale() -> str:
    """Return the active locale, falling back to :data:`DEFAULT_LOCALE`."""
    if _locale_resolver is not None:
        try:
            candidate = _locale_resolver()
        except Exception:  # noqa: BLE001 - a missing session must never break rendering
            candidate = None
        if candidate in SUPPORTED_LOCALES:
            return candidate
    return DEFAULT_LOCALE


def _lookup(locale: str, key: str) -> str | None:
    return _CATALOGS.get(locale, {}).get(key)


def t(key: str, *, locale: str | None = None, **fmt: Any) -> str:
    """Translate ``key`` into the active (or explicit) locale.

    Falls back to the default locale and, ultimately, to ``key`` itself. Extra
    keyword arguments are interpolated with :meth:`str.format`; unknown
    placeholders are left untouched instead of raising.
    """
    active = locale if is_supported_locale(locale) else get_locale()
    text = _lookup(active, key)
    if text is None and active != DEFAULT_LOCALE:
        text = _lookup(DEFAULT_LOCALE, key)
    if text is None:
        text = key
    if fmt:
        try:
            text = text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            pass
    return text


def title_case_slug(slug: str, *, locale: str | None = None) -> str:
    """Turn a ``snake_case`` slug into a human-readable, localized title.

    Each word is capitalized except for locale-specific minor words (articles,
    prepositions and common conjunctions), which stay lowercase unless they head
    the title. The slug itself is never altered for storage; this is presentation
    only.
    """
    active = locale if is_supported_locale(locale) else get_locale()
    minor = _TITLE_CASE_MINOR.get(active, _TITLE_CASE_MINOR[DEFAULT_LOCALE])
    words = slug.replace("_", " ").split()
    if not words:
        return slug
    result: list[str] = []
    for index, word in enumerate(words):
        lowered = word.lower()
        if index != 0 and lowered in minor:
            result.append(lowered)
        else:
            result.append(lowered.capitalize())
    return " ".join(result)


def set_taxonomy_name_provider(provider: Callable[..., str | None] | None) -> None:
    """Install (or clear) the callable that resolves taxonomy display names.

    The provider is called as ``provider(kind, slug, locale, parent_slug)`` and
    returns a label or ``None``. Install a database-backed adapter to source names
    from data; without one, resolution falls back to :func:`title_case_slug`.
    """
    global _taxonomy_name_provider
    _taxonomy_name_provider = provider


def _provider_name(kind: str, slug: str, locale: str, parent_slug: str) -> str | None:
    if _taxonomy_name_provider is None:
        return None
    try:
        value = _taxonomy_name_provider(kind, slug, locale, parent_slug)
    except Exception:  # noqa: BLE001 - a missing name must never break rendering
        return None
    return value or None


def taxonomy_name(
    kind: str,
    slug: str,
    *,
    locale: str | None = None,
    parent_slug: str = "",
) -> str | None:
    """Return the localized display name for a taxonomy entry, or ``None``.

    Resolution order: ``provider(active) → provider(default)``. The slug itself is
    never translated; callers fall back to :func:`title_case_slug` when this
    returns ``None``.
    """
    active = locale if is_supported_locale(locale) else get_locale()
    if active != DEFAULT_LOCALE:
        name = _provider_name(kind, slug, active, parent_slug)
        if name:
            return name
    return _provider_name(kind, slug, DEFAULT_LOCALE, parent_slug)


def course_name(slug: str, *, locale: str | None = None) -> str:
    """Return the localized, human-readable name for a curricular ``slug``.

    Looks the slug up through the taxonomy provider and falls back to
    :func:`title_case_slug` in the same locale, so a name is always returned
    without mixing languages. The slug is never translated.
    """
    active = locale if is_supported_locale(locale) else get_locale()
    known = taxonomy_name("course", slug, locale=active)
    if known:
        return known
    return title_case_slug(slug, locale=active)
