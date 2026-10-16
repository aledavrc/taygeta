"""Copias traducidas por idioma de las preguntas confirmadas.

Cada pregunta conserva su enunciado **canónico** (idioma original) en ``questions``;
estas copias son derivadas y viven en ``question_translations``. Nunca contienen
soluciones ni respuestas correctas, no participan en la deduplicación (el
``canonical_hash`` se calcula solo sobre el enunciado original) y se validan con
la heurística local antes de almacenarse.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from taygeta.core.schema import Choice
from taygeta.i18n import is_supported_locale

__all__ = ["QuestionTranslation"]


class QuestionTranslation(BaseModel):
    """Copia traducida (enunciado y alternativas) de una pregunta, por idioma."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1, description="Id de la pregunta original.")
    locale: str = Field(..., description="Idioma destino (p. ej. ``es`` o ``en``).")
    statement_raw: str = Field(
        ..., min_length=1, description="Enunciado traducido en Markdown/TeX."
    )
    choices: list[Choice] | None = Field(
        default=None, description="Alternativas traducidas (mismos ids y orden que el original)."
    )
    source: str = Field(default="llm", description="Procedencia del nombre (llm/manual).")
    created_at: str = Field(..., description="Fecha/hora de creación en ISO 8601.")

    @field_validator("locale")
    @classmethod
    def _validate_locale(cls, value: str) -> str:
        if not is_supported_locale(value):
            raise ValueError(f"locale no soportado: {value!r}")
        return value
