"""Semantic question deduplication by embeddings (ingestion filter).

Flow:

1. On ingesting a question, its embedding is generated with the configured model
   (:class:`~taygeta.agents.embeddings.Embedder`).
2. It is compared (cosine similarity) against the embeddings already stored in
   the database, restricted to the same course.
3. If any similarity exceeds ``DUPLICATE_SIMILARITY_THRESHOLD``, the question is
   marked as a duplicate candidate.
4. Optionally, an LLM judge decides whether the new question and the candidate
   are mathematically the same question (ignoring notation/OCR variations). If
   no judge is available, the highest-similarity candidate is used.
5. If the duplicate is confirmed, the new text is discarded and the metadata
   (``source_reference`` origins and prerequisites) is merged into the existing
   question. Solutions or correct answers are never stored.

Embedding generation is local (``sentence-transformers``) and the LLM judge
degrade in a controlled way: if the local model or the provider are unavailable,
ingestion continues and only the exact-hash deduplication of
:meth:`Database.insert_question` is kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.schema import EnrichedQuestion
from taygeta.core.settings import Settings, get_settings
from taygeta.server.db import Database

if TYPE_CHECKING:
    import instructor  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "DedupOutcome",
    "DeduplicationService",
    "DuplicateJudgement",
    "merge_duplicate_metadata",
]

#: Maximum number of candidates to send to the LLM judge.
_MAX_CANDIDATES = 5


class DuplicateJudgement(BaseModel):
    """LLM judge verdict on a duplicate candidate."""

    model_config = ConfigDict(extra="forbid")

    is_duplicate: bool
    matching_index: int = Field(
        default=-1,
        ge=-1,
        description="Índice (0-based) de la candidata duplicada, o -1 si no hay duplicado.",
    )
    reason: str = Field(default="", description="Explicación breve en español.")


@dataclass(slots=True)
class DedupOutcome:
    """Result of the deduplication filter on a new question."""

    status: Literal["INSERTED", "DUPLICATE"]
    existing_id: str | None = None
    similarity: float | None = None
    reason: str = ""


_DUPLICATE_JUDGE_SYSTEM_PROMPT = """Eres un juez de deduplicación de preguntas de \
exámenes científicos. Compara la pregunta nueva con las candidatas y determina si \
es MATEMÁTICAMENTE la misma pregunta que alguna de ellas. Ignora diferencias menores \
de notación (variables renombradas como X vs E, cambios de letra o símbolo), errores \
de transcripción OCR y variaciones de redacción que no alteren el enunciado \
matemático. NO resuelvas el ejercicio y NO evalúes respuestas correctas.

Devuelve:
- is_duplicate: true SOLO si la pregunta nueva es la misma pregunta matemática que \
alguna candidata.
- matching_index: el índice (0-based) de la candidata duplicada, o -1 si no hay duplicado.
- reason: breve explicación en español.
"""


def _origin_references(question: EnrichedQuestion) -> list[str]:
    """Retrieves the known origins (``source_reference``) of a question."""
    metadata = question.source_metadata or {}
    references = [str(ref) for ref in metadata.get("source_references", []) if ref]
    primary = metadata.get("source_reference")
    if primary and str(primary) not in references:
        references.insert(0, str(primary))
    return references


def merge_duplicate_metadata(
    existing: EnrichedQuestion, incoming: EnrichedQuestion
) -> EnrichedQuestion:
    """Merges the metadata of an incoming duplicate question into the existing one.

    Keeps the statement, choices and classification of the already-stored
    question (the "canonical" one) and adds the origins and prerequisites of the
    new one. It does not modify solutions or correct answers (the schema does not
    contain them).
    """
    references = list(dict.fromkeys(_origin_references(existing) + _origin_references(incoming)))
    source_metadata = dict(existing.source_metadata)
    if references:
        source_metadata["source_reference"] = references[0]
        source_metadata["source_references"] = references
    prerequisites = list(dict.fromkeys([*existing.prerequisites, *incoming.prerequisites]))
    return existing.model_copy(
        update={
            "source_metadata": source_metadata,
            "prerequisites": prerequisites,
        }
    )


class DeduplicationService:
    """Coordinates embeddings, similarity and the LLM judge for semantic deduplication."""

    def __init__(
        self,
        database: Database,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        judge_model: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Creates the service.

        Args:
            database: Database access (questions and embeddings).
            settings: Configuration to use; defaults to the global one.
            embedder: An already-built :class:`Embedder` (injectable in tests).
            judge_model: Overrides ``LLM_MODEL`` for the duplicate judge.
            threshold: Overrides ``DUPLICATE_SIMILARITY_THRESHOLD`` (useful in
                migration scripts and tests).
        """
        self.settings = settings or get_settings()
        self.database = database
        self._embedder = embedder
        self._judge_model = judge_model
        self._threshold = threshold
        self._judge_client: Any | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    @property
    def threshold(self) -> float:
        if self._threshold is not None:
            return self._threshold
        return self.settings.DUPLICATE_SIMILARITY_THRESHOLD

    @property
    def _judge_model_name(self) -> str:
        return self._judge_model or self.settings.LLM_MODEL

    @property
    def _judge_client_instance(self) -> Any:
        if self._judge_client is None:
            import instructor

            self._judge_client = instructor.from_openai(
                OpenAI(
                    base_url=self.settings.LLM_BASE_URL,
                    api_key=self.settings.LLM_API_KEY,
                    timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
                )
            )
        return self._judge_client

    def embed_question(self, question: EnrichedQuestion) -> list[float] | None:
        """Generates a question embedding on a best-effort basis.

        Returns ``None`` (without raising) if the local model fails to load or
        encode, so as not to block ingestion.
        """
        try:
            return self.embedder.embed(question_embedding_text(question))
        except Exception:  # noqa: BLE001 - semantic deduplication is optional
            logger.warning(
                "could not generate the embedding for question %s", question.id, exc_info=True
            )
            return None

    def ingest(self, question: EnrichedQuestion) -> DedupOutcome:
        """Runs the ingestion filter and returns the result.

        Generates the embedding, looks up candidates by similarity, queries the
        judge and, if a duplicate is confirmed, merges the metadata into the
        existing question. Otherwise it delegates to
        :meth:`Database.insert_question`.
        """
        embedding = self.embed_question(question)
        if embedding is not None:
            matches = self.database.find_similar_questions(
                embedding,
                threshold=self.threshold,
                course=question.course,
                limit=_MAX_CANDIDATES,
            )
            if matches:
                candidates, similarities, ids = self._load_candidates(matches)
                if candidates:
                    match_index = self._resolve_candidate(question, candidates)
                    if match_index is not None:
                        existing_id = ids[match_index]
                        self._merge_into(existing_id, question)
                        return DedupOutcome(
                            status="DUPLICATE",
                            existing_id=existing_id,
                            similarity=similarities[match_index],
                            reason="duplicate question (semantic similarity)",
                        )
        status = self.database.insert_question(question, embedding=embedding)
        return DedupOutcome(status=status)

    def judge(
        self, new_question: EnrichedQuestion, candidates: list[EnrichedQuestion]
    ) -> DuplicateJudgement | None:
        """Asks the LLM judge whether the new question duplicates any candidate.

        Returns ``None`` if there are no candidates, the LLM is not configured or
        the judge fails; in that case the caller can fall back to the similarity
        threshold.
        """
        if not candidates or not self.settings.llm_configured:
            return None
        messages = [
            {"role": "system", "content": _DUPLICATE_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_judge_user_message(new_question, candidates)},
        ]
        try:
            return self._judge_client_instance.chat.completions.create(
                model=self._judge_model_name,
                response_model=DuplicateJudgement,
                messages=messages,
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001 - the judge is optional; fall back to the threshold
            logger.warning(
                "the duplicate judge failed; falling back to the similarity threshold",
                exc_info=True,
            )
            return None

    def _load_candidates(
        self, matches: list[tuple[str, float]]
    ) -> tuple[list[EnrichedQuestion], list[float], list[str]]:
        candidates: list[EnrichedQuestion] = []
        similarities: list[float] = []
        ids: list[str] = []
        for question_id, similarity in matches:
            candidate = self.database.get_question(question_id)
            if candidate is None:
                continue
            candidates.append(candidate)
            similarities.append(similarity)
            ids.append(question_id)
        return candidates, similarities, ids

    def _resolve_candidate(
        self, new_question: EnrichedQuestion, candidates: list[EnrichedQuestion]
    ) -> int | None:
        judgement = self.judge(new_question, candidates)
        if judgement is not None:
            if judgement.is_duplicate and 0 <= judgement.matching_index < len(candidates):
                return judgement.matching_index
            return None
        # No judge (not configured or failed): the highest-similarity candidate.
        return 0

    def _merge_into(self, existing_id: str, incoming: EnrichedQuestion) -> None:
        existing = self.database.get_question(existing_id)
        if existing is None:
            return
        merged = merge_duplicate_metadata(existing, incoming)
        # The statement does not change, so the existing embedding is still valid.
        self.database.update_question(merged)

    @staticmethod
    def _build_judge_user_message(
        new_question: EnrichedQuestion, candidates: list[EnrichedQuestion]
    ) -> str:
        lines = ["Pregunta nueva a evaluar (Markdown/TeX):", "", new_question.statement_raw]
        if new_question.choices:
            lines += ["", "Alternativas:"]
            lines += [f"- {choice.id}: {choice.content_raw}" for choice in new_question.choices]
        lines += ["", "Candidatas ya existentes:"]
        for index, candidate in enumerate(candidates):
            lines += [f"[{index}] {candidate.statement_raw}"]
            if candidate.choices:
                alternatives = " | ".join(
                    f"{choice.id}: {choice.content_raw}" for choice in candidate.choices
                )
                lines.append(f"    Alternativas: {alternatives}")
        lines += [
            "",
            "Decide si la pregunta nueva es MATEMÁTICAMENTE la misma que alguna de las "
            "candidatas (ignorando diferencias menores de notación, OCR o redacción). "
            "Devuelve is_duplicate y matching_index.",
        ]
        return "\n".join(lines)
