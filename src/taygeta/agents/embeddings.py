"""100% local embedding generation via ``sentence-transformers``.

The embeddings for semantic deduplication are computed with a
``sentence-transformers`` model run locally in Python (no calls to an external
service or a cloud embedding API). The model identifier is configured with
``LLM_EMBEDDING_MODEL`` (a Hugging Face id or a local path) and its weights are
downloaded/cached the first time they are used, just like the OCR weights.

The vectorized text is the visible content of the statement and its choices; it
never includes solutions or correct answers (the schema does not contain them).
The vectors are normalized (``normalize_embeddings=True``) so that the cosine
similarity is directly the dot product.

If the model cannot be loaded (no weights, no network the first time, etc.),
:class:`Embedder` raises :class:`EmbeddingError` and callers must degrade in a
controlled way: semantic deduplication is disabled without blocking ingestion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from taygeta.core.settings import Settings, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

__all__ = ["Embedder", "EmbeddingError", "question_embedding_text"]


class EmbeddingError(RuntimeError):
    """Controlled error when loading the local model or generating embeddings."""


def question_embedding_text(question: Any) -> str:
    """Compose the representative text of a question for vectorization.

    It includes the statement and, if present, the choices. It does not include a
    solution or correct answer: only the visible content of the question.
    """
    parts = [question.statement_raw]
    for choice in question.choices or []:
        parts.append(choice.content_raw)
    return "\n".join(parts)


class Embedder:
    """Local embedding generator with ``sentence-transformers``.

    The model is instantiated lazily in :meth:`model` so that weights are not
    downloaded and ``torch``/``sentence_transformers`` are not imported until a
    vector is actually needed.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Create the embedding generator.

        Args:
            settings: Configuration to use; the global one by default.
            model_name: Overrides ``LLM_EMBEDDING_MODEL``.
            model: An already loaded ``SentenceTransformer`` instance (injectable
                in tests to avoid downloading weights).
        """
        self.settings = settings or get_settings()
        self._model_name = model_name
        self._model: SentenceTransformer | None = model

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model in use."""
        return self._model_name or self.settings.LLM_EMBEDDING_MODEL

    @property
    def model(self) -> SentenceTransformer:
        """Load (and cache) the configured ``SentenceTransformer`` model.

        The import and the weight download are deferred until the first call; if
        the model is not available, the underlying exception propagates and the
        caller must degrade in a controlled way.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate the normalized embedding vector of a text.

        Raises:
            EmbeddingError: if the model does not produce a vector.
        """
        try:
            vector = self.model.encode(text, normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001 - local model failure
            raise EmbeddingError(f"failed to generate the embedding: {exc}") from exc
        if vector is None or len(vector) == 0:
            raise EmbeddingError("the model returned no embedding vector")
        return [float(x) for x in vector]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate the embedding vectors of a list of texts, in batches.

        Returns a list of the same size as ``texts``.
        """
        if not texts:
            return []
        try:
            vectors = self.model.encode(list(texts), normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001 - local model failure
            raise EmbeddingError(f"failed to generate the embeddings: {exc}") from exc
        return [[float(x) for x in vector] for vector in vectors]
