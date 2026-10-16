"""Numeric helpers for embedding-based semantic deduplication.

Taygeta computes cosine similarity in memory (pure Python, no extra
dependencies) because the question volume is low. Vectors are stored in SQLite as
JSON text in the ``questions.embedding_json`` column.

None of these functions handle solutions or correct answers: they operate solely
on numeric vectors derived from the visible content of the statement and its
choices.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

__all__ = ["cosine_similarity", "decode_embedding", "encode_embedding"]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity between two vectors, in the range [-1, 1].

    If either vector is empty, they differ in length or have zero norm, returns
    ``0.0`` (no similarity) instead of raising an error.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def encode_embedding(vector: Sequence[float] | None) -> str | None:
    """Serialize an embedding vector as JSON text, or ``None`` if it is empty."""
    if not vector:
        return None
    return json.dumps([float(x) for x in vector])


def decode_embedding(value: str | None) -> list[float]:
    """Deserialize a stored JSON embedding; returns ``[]`` if it is invalid."""
    if not value:
        return []
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [float(x) for x in data if isinstance(x, (int, float))]
