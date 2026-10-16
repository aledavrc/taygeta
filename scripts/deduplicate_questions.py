#!/usr/bin/env python3
"""Semantic deduplication of already-stored questions (one-off migration).

This script:

1. generates the embedding of every question that does not have one yet (with
   ``LLM_EMBEDDING_MODEL``) and persists it in ``questions.embedding_json``;
2. computes the cosine similarity between all questions of the same course;
3. for each pair whose similarity exceeds the threshold, optionally asks the
   LLM judge to confirm that they are mathematically the same question;
4. merges the confirmed duplicates: keeps the oldest question, adds the new
   one's ``source_reference`` and prerequisites, and deletes the redundant row.

No solutions or correct answers are ever discarded (the database does not
contain them).

Usage:

    uv run python scripts/deduplicate_questions.py --db /path/to/taygeta.db
    uv run python scripts/deduplicate_questions.py --db /path/to/taygeta.db --dry-run
    uv run python scripts/deduplicate_questions.py --threshold 0.95 --no-judge --db taygeta.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

from taygeta.agents.embeddings import Embedder, question_embedding_text
from taygeta.core.embedding import cosine_similarity
from taygeta.core.schema import EnrichedQuestion
from taygeta.server.db import Database
from taygeta.server.deduplication import DeduplicationService, merge_duplicate_metadata

logger = logging.getLogger("deduplicate_questions")


def _resolve_db_path(explicit: str | None) -> str:
    """Resolve the database path: explicit argument or configuration."""
    if explicit:
        return explicit
    try:
        from taygeta.core.settings import get_settings

        return get_settings().database_path
    except Exception:  # pragma: no cover - no configuration available
        return "/app/data/taygeta.db"


def _load_questions(database: Database) -> list[EnrichedQuestion]:
    """Load all questions from the database, sorted by creation date."""
    questions: list[EnrichedQuestion] = []
    for question_id in database.list_question_ids():
        question = database.get_question(question_id)
        if question is not None:
            questions.append(question)
    questions.sort(key=lambda q: (q.created_at, q.id))
    return questions


def backfill_embeddings(database: Database, embedder: Embedder, *, dry_run: bool = False) -> int:
    """Generate and persist the missing embeddings. Returns how many it created."""
    questions = _load_questions(database)
    missing = [q for q in questions if not database.get_embedding(q.id)]
    if not missing:
        return 0

    texts = [question_embedding_text(q) for q in missing]
    try:
        vectors = embedder.embed_many(texts)
    except Exception as exc:  # noqa: BLE001 - the migration must not abort because of the provider
        logger.warning("could not generate embeddings in bulk: %s", exc)
        vectors = []
        for text in texts:
            try:
                vectors.append(embedder.embed(text))
            except Exception as item_exc:  # noqa: BLE001 - degrades per item
                logger.warning("individual embedding skipped: %s", item_exc)
                vectors.append([])

    created = 0
    for question, vector in zip(missing, vectors, strict=True):
        if not vector:
            continue
        if not dry_run:
            database.update_question(question, embedding=vector)
        created += 1
    return created


def deduplicate(
    database: Database,
    *,
    threshold: float,
    use_judge: bool,
    dry_run: bool = False,
) -> int:
    """Merge semantic duplicates and return how many it merged.

    Compares within the same course and keeps the oldest question.
    """
    service = DeduplicationService(database, threshold=threshold)
    questions = _load_questions(database)

    by_course: dict[str, list[EnrichedQuestion]] = defaultdict(list)
    for question in questions:
        by_course[question.course].append(question)

    merged_ids: set[str] = set()
    merged = 0
    for course_questions in by_course.values():
        for index, question in enumerate(course_questions):
            if question.id in merged_ids:
                continue
            embedding = database.get_embedding(question.id)
            if not embedding:
                continue
            for other in course_questions[index + 1 :]:
                if other.id in merged_ids:
                    continue
                other_embedding = database.get_embedding(other.id)
                if not other_embedding:
                    continue
                similarity = cosine_similarity(embedding, other_embedding)
                if similarity < threshold:
                    continue
                if not _is_confirmed_duplicate(service, question, other, use_judge):
                    continue
                merged_question = merge_duplicate_metadata(question, other)
                if not dry_run:
                    database.update_question(merged_question, embedding=embedding)
                    database.delete_question(other.id)
                merged += 1
                merged_ids.add(other.id)
    return merged


def _is_confirmed_duplicate(
    service: DeduplicationService,
    question: EnrichedQuestion,
    other: EnrichedQuestion,
    use_judge: bool,
) -> bool:
    """Confirm (by LLM judge or by threshold) that ``other`` is a duplicate of ``question``."""
    if not use_judge:
        return True
    judgement = service.judge(question, [other])
    if judgement is None:
        # Judge unavailable or failed: the similarity threshold is respected.
        return True
    return judgement.is_duplicate and judgement.matching_index == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and merge semantically duplicated questions."
    )
    parser.add_argument("--db", default=None, help="Path to the SQLite file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold (defaults to DUPLICATE_SIMILARITY_THRESHOLD).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Do not consult the LLM judge; merge only by similarity threshold.",
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Do not generate embeddings for questions that do not have them yet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only reports the merges and embeddings that would be performed, without writing.",
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    try:
        database = Database(db_path)
        settings = database.settings
        threshold = (
            args.threshold
            if args.threshold is not None
            else settings.DUPLICATE_SIMILARITY_THRESHOLD
        )
        if not 0.0 <= threshold <= 1.0:
            print("the threshold must be between 0.0 and 1.0", file=sys.stderr)
            raise SystemExit(2)

        created = 0
        if not args.no_backfill:
            created = backfill_embeddings(database, Embedder(settings), dry_run=args.dry_run)

        merged = deduplicate(
            database,
            threshold=threshold,
            use_judge=not args.no_judge,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - reported and exited with an error
        print(f"error processing the database {db_path!r}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    dry_run_note = " (dry run)" if args.dry_run else ""
    print(
        f"{created} embedding(s) generated and {merged} duplicate question(s) "
        f"merged in {db_path}{dry_run_note} "
        f"(threshold={threshold}, judge={'no' if args.no_judge else 'yes'})."
    )


if __name__ == "__main__":
    main()
