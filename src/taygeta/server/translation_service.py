"""Translation orchestration for confirmed questions.

Translates a finalized question into every supported locale, validates each copy
with the local heuristic layer and stores the valid ones. It is best-effort: a
translation failure is reported, never hidden and never blocks confirmation of
the canonical question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from taygeta.agents.translator import Translator
from taygeta.core.schema import EnrichedQuestion
from taygeta.ingest.validator import validate_translation
from taygeta.server.db import Database

logger = logging.getLogger(__name__)

__all__ = ["TranslationOutcome", "TranslationService"]


@dataclass(slots=True)
class TranslationOutcome:
    """Resultado de traducir y almacenar las copias de una pregunta."""

    stored: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class TranslationService:
    """Traduce, valida y almacena las copias localizadas de una pregunta.

    Cada copia pasa la heurística local (:func:`validate_translation`) y, si se
    inyecta un ``semantic`` validador, también la capa semántica remota (que
    verifica, entre otras cosas, que no se hayan añadido soluciones). Una copia
    que no supere la validación se descarta y se reporta; nunca se almacena.
    """

    def __init__(
        self,
        database: Database,
        translator: Translator | None = None,
        *,
        semantic: Any | None = None,
    ) -> None:
        self.database = database
        self.translator = translator or Translator()
        self.semantic = semantic

    def store(self, question: EnrichedQuestion) -> TranslationOutcome:
        outcome = TranslationOutcome()
        try:
            translations = self.translator.translate(question)
        except Exception as exc:  # noqa: BLE001 - la traducción es complementaria
            logger.warning("no se pudo traducir la pregunta %s: %s", question.id, exc)
            outcome.problems.append(f"{question.id}: no se pudo traducir ({exc})")
            return outcome
        for translation in translations:
            result = validate_translation(question, translation)
            if not result.is_valid:
                logger.warning(
                    "la traducción %s de %s no superó la validación: %s",
                    translation.locale,
                    question.id,
                    result.format_reasons(),
                )
                outcome.problems.append(
                    f"{question.id} [{translation.locale}]: {result.format_reasons()}"
                )
                continue
            if not self._semantic_ok(question, translation):
                logger.warning(
                    "la traducción %s de %s no superó la validación semántica",
                    translation.locale,
                    question.id,
                )
                outcome.problems.append(
                    f"{question.id} [{translation.locale}]: no superó la validación semántica"
                )
                continue
            try:
                self.database.upsert_question_translation(translation)
            except Exception as exc:  # noqa: BLE001 - se informa, no se oculta
                outcome.problems.append(f"{question.id} [{translation.locale}]: {exc}")
                continue
            outcome.stored.append(translation.locale)
        return outcome

    def _semantic_ok(self, question: EnrichedQuestion, translation: Any) -> bool:
        """Comprueba la copia con la capa semántica, si está inyectada.

        Un fallo del proveedor se trata como no verificable y descarta la copia,
        de modo que nunca se almacena una traducción sin validar.
        """
        if self.semantic is None:
            return True
        candidate = question.model_copy(
            update={
                "statement_raw": translation.statement_raw,
                "choices": translation.choices,
            }
        )
        try:
            logical = self.semantic.validate(candidate)
        except Exception as exc:  # noqa: BLE001 - capa remota opcional
            logger.warning("validación semántica no disponible: %s", exc)
            return False
        return bool(logical.is_valid)
