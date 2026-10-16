"""Seed rows for the localized taxonomy labels (initial ``es``/``en`` course names).

Pure data: consumed once by :meth:`taygeta.server.db.Database.initialize` to
populate ``taxonomy_labels`` on a fresh database. After seeding, the database is
the source of truth — these values are only a bootstrap for the known courses, so
future additions are data inserts rather than code edits.
"""

from __future__ import annotations

#: Curricular slug -> ``(spanish_name, english_name)``.
_COURSE_NAMES: dict[str, tuple[str, str]] = {
    "analisis_funcional": ("Análisis Funcional", "Functional Analysis"),
    "analisis_real": ("Análisis Real", "Real Analysis"),
    "analisis_complejo": ("Análisis Complejo", "Complex Analysis"),
    "calculo": ("Cálculo", "Calculus"),
    "calculo_diferencial": ("Cálculo Diferencial", "Differential Calculus"),
    "calculo_integral": ("Cálculo Integral", "Integral Calculus"),
    "ecuaciones_diferenciales": ("Ecuaciones Diferenciales", "Differential Equations"),
    "algebra": ("Álgebra", "Algebra"),
    "algebra_lineal": ("Álgebra Lineal", "Linear Algebra"),
    "algebra_abstracta": ("Álgebra Abstracta", "Abstract Algebra"),
    "topologia": ("Topología", "Topology"),
    "geometria": ("Geometría", "Geometry"),
    "probabilidad": ("Probabilidad", "Probability"),
    "estadistica": ("Estadística", "Statistics"),
    "variable_compleja": ("Variable Compleja", "Complex Variables"),
}


def _build_seed() -> tuple[tuple[str, str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str, str]] = []
    for slug, (spanish, english) in _COURSE_NAMES.items():
        rows.append(("course", "", slug, "es", spanish))
        rows.append(("course", "", slug, "en", english))
    return tuple(rows)


#: ``(kind, parent_slug, slug, locale, label)`` rows seeded with ``source='seed'``.
SEED_LABELS: tuple[tuple[str, str, str, str, str], ...] = _build_seed()
