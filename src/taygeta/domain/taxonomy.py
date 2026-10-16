"""Taxonomía curricular de Taygeta: nombres legibles de curso.

Helpers de presentación del dominio curricular, que delegan en la capa de
internacionalización (:mod:`taygeta.i18n`):

* :func:`title_case_slug`: slug ``snake_case`` → título legible en el idioma
  activo (con su propio conjunto de palabras menores).
* :func:`format_course_name`: nombre de curso localizado al idioma activo,
  resuelto a través del proveedor de taxonomía (base de datos) con respaldo en
  :func:`title_case_slug`.
* :class:`TaxonomyNames` y :func:`filter_taxonomy_names`: transportan los nombres
  legibles (es/en) que genera el LLM durante la estructuración hasta la
  confirmación, donde se persisten con ``source='llm'``.

Los slugs curriculares siguen siendo **datos estables** y nunca se traducen; solo
varía el nombre de presentación. Los nombres legibles viven en la tabla
``taxonomy_labels`` (no en el código).

La canonicalización de ``course``, ``topic``, ``subtopic`` y prerrequisitos es
100 % dinámica: se lee de la base de datos en vivo (ver
:meth:`taygeta.server.db.Database.get_live_taxonomy_tree` y
:meth:`taygeta.server.db.Database.list_prerequisites`) y se inyecta en los system
prompts de estructuración. No hay diccionarios de alias ni vocabulario cerrado
hardcodeado.

Ninguno de estos helpers almacena soluciones ni respuestas correctas: son
únicamente nombres de presentación.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from taygeta.core.schema import Slug
from taygeta.i18n import course_name, title_case_slug

__all__ = [
    "TAXONOMY_KINDS",
    "TaxonomyKind",
    "TaxonomyNames",
    "filter_taxonomy_names",
    "format_course_name",
    "title_case_slug",
]

#: Tipos de entrada de la taxonomía que pueden tener un nombre legible.
TaxonomyKind = Literal["course", "topic", "subtopic", "prerequisite"]

TAXONOMY_KINDS: tuple[TaxonomyKind, ...] = ("course", "topic", "subtopic", "prerequisite")


class TaxonomyNames(BaseModel):
    """Nombres legibles (es/en) de una entrada de taxonomía.

    Viaja desde la estructuración (generados por el LLM) hasta la confirmación,
    donde se persisten en ``taxonomy_labels`` con ``source='llm'``. Los nombres
    son solo de presentación: nunca sustituyen al slug ni contienen soluciones.
    """

    model_config = ConfigDict(extra="forbid")

    kind: TaxonomyKind
    slug: Slug
    parent_slug: str = ""
    es: str = Field(..., min_length=1, description="Nombre legible en español.")
    en: str = Field(..., min_length=1, description="Nombre legible en inglés.")

    def locale_labels(self) -> list[tuple[str, str]]:
        """Devuelve los pares ``(locale, label)`` listos para persistir."""
        return [("es", self.es), ("en", self.en)]


def filter_taxonomy_names(
    names: Iterable[TaxonomyNames],
    *,
    course: str,
    topic: str,
    subtopic: str | None,
    prerequisites: Iterable[str],
) -> list[TaxonomyNames]:
    """Conserva solo los nombres que aún coinciden con la clasificación final.

    La revisión humana puede editar curso, tema, subtema o prerrequisitos, por lo
    que los nombres generados por el LLM que ya no correspondan a esos slugs se
    descartan. Los temas y subtemas se re-vinculan al curso final.
    """
    prerequisite_set = set(prerequisites)
    kept: list[TaxonomyNames] = []
    for entry in names:
        if entry.kind == "course":
            if entry.slug == course:
                kept.append(entry)
        elif entry.kind == "topic":
            if entry.slug == topic:
                kept.append(entry.model_copy(update={"parent_slug": course}))
        elif entry.kind == "subtopic":
            if subtopic and entry.slug == subtopic:
                kept.append(entry.model_copy(update={"parent_slug": course}))
        elif entry.kind == "prerequisite" and entry.slug in prerequisite_set:
            kept.append(entry.model_copy(update={"parent_slug": ""}))
    return kept


def format_course_name(slug: str, *, locale: str | None = None) -> str:
    """Devuelve el nombre legible y localizado de un curso.

    Resuelve el slug a través del proveedor de taxonomía (típicamente respaldado
    por la base de datos) y aplica :func:`title_case_slug` como respaldo para
    slugs sin nombre catalogado.
    """
    return course_name(slug, locale=locale)
