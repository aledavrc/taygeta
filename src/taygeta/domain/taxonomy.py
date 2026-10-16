"""Taxonomía curricular de Taygeta: nombres legibles de curso.

Helpers de presentación del dominio curricular, que delegan en la capa de
internacionalización (:mod:`taygeta.i18n`):

* :func:`title_case_slug`: slug ``snake_case`` → título legible en el idioma
  activo (con su propio conjunto de palabras menores).
* :func:`format_course_name`: nombre de curso localizado al idioma activo,
  resuelto a través del proveedor de taxonomía (base de datos) con respaldo en
  :func:`title_case_slug`.

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

from taygeta.i18n import course_name, title_case_slug

__all__ = [
    "format_course_name",
    "title_case_slug",
]


def format_course_name(slug: str, *, locale: str | None = None) -> str:
    """Devuelve el nombre legible y localizado de un curso.

    Resuelve el slug a través del proveedor de taxonomía (típicamente respaldado
    por la base de datos) y aplica :func:`title_case_slug` como respaldo para
    slugs sin nombre catalogado.
    """
    return course_name(slug, locale=locale)
