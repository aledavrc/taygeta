"""Per-language translated copies of confirmed questions.

Each question keeps its **canonical** statement (original language) in
``questions``; these copies are derived and live in ``question_translations``.
They never contain solutions or correct answers, they do not participate in
deduplication (``canonical_hash`` is computed only over the original statement)
and they are validated with the local heuristic before being stored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from taygeta.core.schema import Choice
from taygeta.i18n import is_supported_locale

__all__ = ["QuestionTranslation"]


class QuestionTranslation(BaseModel):
    """Translated copy (statement and choices) of a question, per language."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1, description="Id of the original question.")
    locale: str = Field(..., description="Target language (e.g. ``es`` or ``en``).")
    statement_raw: str = Field(
        ..., min_length=1, description="Translated statement in Markdown/TeX."
    )
    choices: list[Choice] | None = Field(
        default=None, description="Translated choices (same ids and order as the original)."
    )
    source: str = Field(default="llm", description="Origin of the name (llm/manual).")
    created_at: str = Field(..., description="Creation date/time in ISO 8601.")

    @field_validator("locale")
    @classmethod
    def _validate_locale(cls, value: str) -> str:
        if not is_supported_locale(value):
            raise ValueError(f"unsupported locale: {value!r}")
        return value
