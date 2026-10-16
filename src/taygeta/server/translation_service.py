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
    """Outcome of translating and storing the copies of a question."""

    stored: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class TranslationService:
    """Translates, validates and stores the localized copies of a question.

    Each copy goes through the local heuristic (:func:`validate_translation`)
    and, when a ``semantic`` validator is injected, also the remote semantic
    layer (which checks, among other things, that no solutions were added). A
    copy that fails validation is dropped and reported; it is never stored.
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
        except Exception as exc:  # noqa: BLE001 - translation is complementary
            logger.warning("could not translate question %s: %s", question.id, exc)
            outcome.problems.append(f"{question.id}: could not translate ({exc})")
            return outcome
        for translation in translations:
            result = validate_translation(question, translation)
            if not result.is_valid:
                logger.warning(
                    "translation %s of %s did not pass validation: %s",
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
                    "translation %s of %s did not pass semantic validation",
                    translation.locale,
                    question.id,
                )
                outcome.problems.append(
                    f"{question.id} [{translation.locale}]: did not pass semantic validation"
                )
                continue
            try:
                self.database.upsert_question_translation(translation)
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                outcome.problems.append(f"{question.id} [{translation.locale}]: {exc}")
                continue
            outcome.stored.append(translation.locale)
        return outcome

    def _semantic_ok(self, question: EnrichedQuestion, translation: Any) -> bool:
        """Checks the copy with the semantic layer, if one is injected.

        A provider failure is treated as unverifiable and drops the copy, so an
        unvalidated translation is never stored.
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
        except Exception as exc:  # noqa: BLE001 - optional remote layer
            logger.warning("semantic validation unavailable: %s", exc)
            return False
        return bool(logical.is_valid)
