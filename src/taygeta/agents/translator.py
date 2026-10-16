"""Question translator agent: localized copies (es/en) of a confirmed question.

Runs a single remote LLM call per question and returns one
:class:`~taygeta.domain.translation.QuestionTranslation` per supported locale.
The translation must keep all math (TeX) and ``{{fig:N}}`` placeholders exactly;
each copy is validated locally before storage (see
:func:`taygeta.ingest.validator.validate_translation`). It never adds solutions
or answers, and it never touches the canonical statement used for deduplication.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from taygeta.core.schema import (
    Choice,
    EnrichedQuestion,
    sanitize_math_delimiters,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.translation import QuestionTranslation
from taygeta.i18n import SUPPORTED_LOCALES

logger = logging.getLogger(__name__)

__all__ = [
    "TranslationBundle",
    "TranslationDraft",
    "TranslationError",
    "Translator",
]

_TRANSLATION_SYSTEM_PROMPT = """You translate scientific exam questions. You \
receive a question statement and, when present, its answer choices. Return a \
faithful translation for EACH requested locale. Rules:

1. Keep every mathematical expression byte-for-byte identical, both inline \
($...$) and display ($$...$$). Never translate, simplify, reformat or reorder math.
2. Keep every {{fig:N}} placeholder exactly where it is; never add, remove or move \
a placeholder.
3. Translate only the natural-language prose. Keep the same number of choices, the \
same choice ids and the same order.
4. NEVER add solutions, answers, hints, explanations or any content that is not \
present in the source. Do not answer the question.
5. Provide one translation entry per requested locale (Spanish and English).
"""


class TranslationDraft(BaseModel):
    """Translated statement (and choices) for a single locale."""

    model_config = ConfigDict(extra="forbid")

    statement_raw: str = Field(
        ..., min_length=1, description="Translated statement (Markdown/TeX)."
    )
    choices: list[Choice] | None = Field(
        default=None, description="Translated choices, same ids and order as the source."
    )


class TranslationBundle(BaseModel):
    """LLM response: one translated copy per supported locale."""

    model_config = ConfigDict(extra="forbid")

    es: TranslationDraft
    en: TranslationDraft


class TranslationError(RuntimeError):
    """Controlled error raised when the remote translation fails."""


class Translator:
    """Generates localized copies of a question through the remote LLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Create the translator.

        Args:
            settings: Configuration to use; defaults to the global one.
            client: Pre-built ``instructor`` client (injectable for tests).
            model: Overrides ``LLM_MODEL``.
            max_retries: Overrides ``LLM_MAX_RETRIES``.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self._max_retries = (
            max_retries if max_retries is not None else self.settings.LLM_MAX_RETRIES
        )

    @property
    def model(self) -> str:
        return self._model or self.settings.LLM_MODEL

    @property
    def client(self) -> Any:
        if self._client is None:
            import instructor

            self._client = instructor.from_openai(self._openai_client())
        return self._client

    def _openai_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.settings.LLM_BASE_URL,
            api_key=self.settings.LLM_API_KEY,
            timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
            max_retries=self._max_retries,
        )

    def translate(self, question: EnrichedQuestion) -> list[QuestionTranslation]:
        """Translate ``question`` into every supported locale.

        Raises:
            TranslationError: if the provider fails or returns nothing.
        """
        messages = [
            {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(question)},
        ]
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                response_model=TranslationBundle,
                messages=messages,
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - provider and schema errors
            raise TranslationError(f"remote translation failed: {exc}") from exc
        if raw is None:
            raise TranslationError("the remote LLM produced no translation")
        return self._assemble(question, raw)

    @staticmethod
    def _build_user_message(question: EnrichedQuestion) -> str:
        lines = [
            f"Target locales: {', '.join(SUPPORTED_LOCALES)}.",
            "",
            "Statement (Markdown/TeX):",
            question.statement_raw,
        ]
        if question.choices:
            lines += ["", "Choices:"]
            lines += [f"- {choice.id}: {choice.content_raw}" for choice in question.choices]
        if question.figures:
            lines += ["", "Declared figures (placeholders must match):"]
            lines += [f"- {figure.id}" for figure in question.figures]
        return "\n".join(lines)

    @staticmethod
    def _assemble(
        question: EnrichedQuestion, bundle: TranslationBundle
    ) -> list[QuestionTranslation]:
        drafts: dict[str, TranslationDraft | None] = {
            "es": bundle.es,
            "en": bundle.en,
        }
        translations: list[QuestionTranslation] = []
        for locale in SUPPORTED_LOCALES:
            draft = drafts.get(locale)
            if draft is None:
                continue
            translations.append(
                QuestionTranslation(
                    question_id=question.id,
                    locale=locale,
                    statement_raw=sanitize_math_delimiters(draft.statement_raw),
                    choices=draft.choices,
                    source="llm",
                    created_at=utc_now_iso(),
                )
            )
        return translations
