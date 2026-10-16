"""DB-backed taxonomy display names for the i18n layer.

Installs a cached adapter through
:func:`taygeta.i18n.set_taxonomy_name_provider`, so localized
course/topic/subtopic/prerequisite names come from the database
(``taxonomy_labels``) rather than from code. The UI installs it once (see
``app._resources``) and invalidates it after taxonomy writes.
"""

from __future__ import annotations

from typing import Any

from taygeta.i18n import set_taxonomy_name_provider

__all__ = [
    "TaxonomyNameProvider",
    "install_taxonomy_names",
    "invalidate_taxonomy_names",
]


class TaxonomyNameProvider:
    """Caches ``(kind, parent_slug, slug) -> label`` maps per locale."""

    def __init__(self, database: Any) -> None:
        self._database = database
        self._cache: dict[str, dict[tuple[str, str, str], str]] = {}

    def invalidate(self) -> None:
        """Drop all cached locale maps (call after taxonomy writes)."""
        self._cache.clear()

    def _labels(self, locale: str) -> dict[tuple[str, str, str], str]:
        cached = self._cache.get(locale)
        if cached is None:
            cached = self._database.taxonomy_label_map(locale=locale)
            self._cache[locale] = cached
        return cached

    def __call__(self, kind: str, slug: str, locale: str, parent_slug: str = "") -> str | None:
        return self._labels(locale).get((kind, parent_slug, slug))


_provider: TaxonomyNameProvider | None = None


def install_taxonomy_names(database: Any) -> TaxonomyNameProvider:
    """Install a DB-backed taxonomy name provider and return it."""
    global _provider
    _provider = TaxonomyNameProvider(database)
    set_taxonomy_name_provider(_provider)
    return _provider


def invalidate_taxonomy_names() -> None:
    """Drop cached labels after taxonomy writes (safe when not installed)."""
    if _provider is not None:
        _provider.invalidate()
