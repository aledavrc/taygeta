"""SQLite persistence with FTS5 full-text search for Taygeta.

This module implements the database described in section 6 of the
specification:

* ``questions`` with the exact requested fields and ``payload_json`` holding the
  serialized :class:`~taygeta.core.schema.EnrichedQuestion` object.
* ``questions_fts`` (FTS5 virtual) kept in sync through AFTER INSERT/UPDATE/DELETE
  triggers, so that full-text search never becomes stale.
* ``courses``, ``topics``, ``prerequisites`` and ``staging_sessions``.

The database lives in ``DB_PATH`` (by default ``/app/data/taygeta.db``), that is,
in the persistent volume, never in the repository root. Schema initialization is
idempotent and :meth:`Database.status` reports the real state of the database.
Solutions and correct answers are not stored: ``payload_json`` is validated by
schemas that forbid them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from taygeta.core.embedding import cosine_similarity, decode_embedding, encode_embedding
from taygeta.core.schema import compute_canonical_hash, utc_now_iso
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.taxonomy import TaxonomyNames
from taygeta.domain.translation import QuestionTranslation
from taygeta.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, is_supported_locale, title_case_slug
from taygeta.server.taxonomy_seed import SEED_LABELS

if TYPE_CHECKING:
    from taygeta.core.schema import EnrichedQuestion

__all__ = [
    "SCHEMA_VERSION",
    "STAGING_STATUSES",
    "Database",
    "DatabaseError",
    "get_database",
]

#: Logical schema version, recorded in ``schema_meta``.
SCHEMA_VERSION = "3"

#: Accepted statuses for a staging session.
STAGING_STATUSES: tuple[str, ...] = ("VALID", "NEEDS_ATTENTION", "CONFIRMED", "DISCARDED")

#: Maximum result limit to protect the RAG service.
MAX_SEARCH_LIMIT = 100

#: Sources of a taxonomy label, from lowest to highest precedence on write.
TAXONOMY_LABEL_SOURCES: tuple[str, ...] = ("llm", "seed", "manual")

#: Numeric precedence used by the ``ON CONFLICT`` of ``taxonomy_labels``.
_SOURCE_RANK_SQL = (
    "CASE excluded.source WHEN 'manual' THEN 3 WHEN 'seed' THEN 2 ELSE 1 END >= "
    "CASE taxonomy_labels.source WHEN 'manual' THEN 3 WHEN 'seed' THEN 2 ELSE 1 END"
)


class DatabaseError(RuntimeError):
    """Controlled error of database access or integrity."""


#: DDL of the taxonomy tables. They are rebuilt when migrating from the previous
#: schema, in which course names lived in ``courses.name``.
_TAXONOMY_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    slug TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS topics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    course_slug   TEXT NOT NULL REFERENCES courses(slug) ON DELETE CASCADE,
    topic_slug    TEXT NOT NULL,
    subtopic_slug TEXT
);

CREATE TABLE IF NOT EXISTS prerequisites (
    slug        TEXT PRIMARY KEY,
    description TEXT
);

-- Human-readable names by locale and taxonomy kind. The slugs remain stable
-- data; ``source`` orders the precedence manual > seed > llm.
CREATE TABLE IF NOT EXISTS taxonomy_labels (
    kind        TEXT NOT NULL,
    parent_slug TEXT NOT NULL DEFAULT '',
    slug        TEXT NOT NULL,
    locale      TEXT NOT NULL,
    label       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'manual',
    PRIMARY KEY (kind, parent_slug, slug, locale)
);

CREATE INDEX IF NOT EXISTS idx_topics_course ON topics(course_slug);

CREATE INDEX IF NOT EXISTS idx_taxonomy_labels_lookup
    ON taxonomy_labels(kind, parent_slug, slug);

-- The subtopic may be NULL; COALESCE stops several NULLs from being treated as distinct.
CREATE UNIQUE INDEX IF NOT EXISTS uq_topics_course_topic_subtopic
    ON topics(course_slug, topic_slug, COALESCE(subtopic_slug, ''));
"""

_SCHEMA_SQL = (
    """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
    + _TAXONOMY_SQL
    + """
CREATE TABLE IF NOT EXISTS questions (
    id                  TEXT PRIMARY KEY,
    course              TEXT NOT NULL,
    topic               TEXT NOT NULL,
    subtopic            TEXT,
    difficulty          INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    prerequisites_json  TEXT NOT NULL DEFAULT '[]',
    payload_json        TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    embedding_json      TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
    id UNINDEXED,
    statement_raw,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Translated copies by locale. The canonical statement lives in ``questions``;
-- these rows are derived, do not take part in deduplication and are deleted in
-- cascade with their question.
CREATE TABLE IF NOT EXISTS question_translations (
    question_id  TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    locale       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'llm',
    created_at   TEXT NOT NULL,
    PRIMARY KEY (question_id, locale)
);

CREATE INDEX IF NOT EXISTS idx_question_translations_locale
    ON question_translations(locale);

CREATE VIRTUAL TABLE IF NOT EXISTS question_translations_fts USING fts5(
    question_id UNINDEXED,
    locale UNINDEXED,
    statement_raw,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS staging_sessions (
    session_id TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'NEEDS_ATTENTION'
);

CREATE INDEX IF NOT EXISTS idx_questions_course_topic ON questions(course, topic);
CREATE INDEX IF NOT EXISTS idx_questions_subtopic ON questions(subtopic);
CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty);
CREATE INDEX IF NOT EXISTS idx_questions_created_at ON questions(created_at);
CREATE INDEX IF NOT EXISTS idx_staging_status ON staging_sessions(status);
CREATE INDEX IF NOT EXISTS idx_staging_created_at ON staging_sessions(created_at);

-- FTS5 synchronization with questions through triggers.
CREATE TRIGGER IF NOT EXISTS questions_ai AFTER INSERT ON questions BEGIN
    INSERT INTO questions_fts(id, statement_raw)
    VALUES (new.id, json_extract(new.payload_json, '$.statement_raw'));
END;

CREATE TRIGGER IF NOT EXISTS questions_ad AFTER DELETE ON questions BEGIN
    DELETE FROM questions_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS questions_au AFTER UPDATE ON questions BEGIN
    DELETE FROM questions_fts WHERE id = old.id;
    INSERT INTO questions_fts(id, statement_raw)
    VALUES (new.id, json_extract(new.payload_json, '$.statement_raw'));
END;

-- FTS5 synchronization of the translations through triggers.
CREATE TRIGGER IF NOT EXISTS question_translations_ai AFTER INSERT ON question_translations BEGIN
    INSERT INTO question_translations_fts(question_id, locale, statement_raw)
    VALUES (new.question_id, new.locale, json_extract(new.payload_json, '$.statement_raw'));
END;

CREATE TRIGGER IF NOT EXISTS question_translations_ad AFTER DELETE ON question_translations BEGIN
    DELETE FROM question_translations_fts
    WHERE question_id = old.question_id AND locale = old.locale;
END;

CREATE TRIGGER IF NOT EXISTS question_translations_au AFTER UPDATE ON question_translations BEGIN
    DELETE FROM question_translations_fts
    WHERE question_id = old.question_id AND locale = old.locale;
    INSERT INTO question_translations_fts(question_id, locale, statement_raw)
    VALUES (new.question_id, new.locale, json_extract(new.payload_json, '$.statement_raw'));
END;
"""
)


def _build_match_query(query: str) -> str:
    """Turns a free query into a safe MATCH expression for FTS5.

    Each term is quoted so that FTS5 syntax (``AND``/``OR``/``NEAR`` operators,
    quotes, parentheses, ``*``) cannot cause errors or unexpected queries.
    """
    tokens = query.split()
    if not tokens:
        raise DatabaseError("the search query cannot be empty")
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class Database:
    """Transactional access to the Taygeta SQLite database."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        initialize: bool = True,
    ) -> None:
        """Creates the database access.

        Args:
            db_path: Path of the SQLite file. If omitted, ``DB_PATH`` from the
                configuration is used.
            settings: Configuration to use; defaults to the global one.
            initialize: If ``True``, initializes the schema idempotently when the
                instance is built.
        """
        self.settings = settings or get_settings()
        resolved = db_path if db_path is not None else self.settings.database_path
        self.db_path = str(resolved)
        self._memory_connection: sqlite3.Connection | None = None
        self._fts5_enabled = False
        if initialize:
            self.initialize()

    # -- Connection ---------------------------------------------------------
    @property
    def is_memory(self) -> bool:
        """Indicates whether the database resides in memory."""
        return self.db_path == ":memory:"

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Transactional context that commits on exit and rolls back on error."""
        if self.is_memory:
            if self._memory_connection is None:
                self._memory_connection = self._new_connection()
            connection = self._memory_connection
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            return

        connection = self._new_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # -- Initialization -----------------------------------------------------
    def initialize(self) -> None:
        """Creates the schema idempotently (safe to call several times).

        Raises:
            DatabaseError: if the path is not accessible or the schema cannot be created.
        """
        try:
            if not self.is_memory:
                Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                if not self.is_memory:
                    connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA_SQL)
                self._migrate_schema(connection)
                self._seed_taxonomy(connection)
                connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"could not initialize the database at {self.db_path!r}: {exc}"
            ) from exc
        self._fts5_enabled = self._check_fts5()

    def _check_fts5(self) -> bool:
        try:
            with self.connect() as connection:
                connection.execute("SELECT count(*) FROM questions_fts").fetchone()
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Applies incremental migrations to already-existing databases.

        ``CREATE TABLE IF NOT EXISTS`` does not alter pre-existing tables, so the
        columns added in later versions of the schema are created here
        idempotently (for example ``questions.embedding_json``).

        Course names stopped living in ``courses.name`` (they now live in
        ``taxonomy_labels``); if the previous schema is detected, the taxonomy
        tables are rebuilt from scratch. The stored data is disposable, so no
        rows are migrated.
        """
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(questions)").fetchall()
        }
        if "embedding_json" not in columns:
            connection.execute("ALTER TABLE questions ADD COLUMN embedding_json TEXT")

        course_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(courses)").fetchall()
        }
        if "name" in course_columns:
            connection.executescript(
                "DROP TABLE IF EXISTS topics;"
                "DROP TABLE IF EXISTS courses;"
                "DROP TABLE IF EXISTS prerequisites;"
            )
            connection.executescript(_TAXONOMY_SQL)

    def _seed_taxonomy(self, connection: sqlite3.Connection) -> None:
        """Seeds ``taxonomy_labels`` the first time (idempotent, does not overwrite)."""
        row = connection.execute("SELECT count(*) AS c FROM taxonomy_labels").fetchone()
        if row is not None and int(row["c"]) > 0:
            return
        for kind, _parent, slug, _locale, _label in SEED_LABELS:
            if kind == "course":
                connection.execute("INSERT OR IGNORE INTO courses(slug) VALUES (?)", (slug,))
        connection.executemany(
            "INSERT OR IGNORE INTO taxonomy_labels"
            "(kind, parent_slug, slug, locale, label, source) "
            "VALUES (?, ?, ?, ?, ?, 'seed')",
            SEED_LABELS,
        )

    def rebuild_fts(self) -> int:
        """Rebuilds the FTS5 tables and returns the number of question rows."""
        with self.connect() as connection:
            connection.execute("DELETE FROM questions_fts")
            connection.execute(
                "INSERT INTO questions_fts(id, statement_raw) "
                "SELECT id, json_extract(payload_json, '$.statement_raw') FROM questions"
            )
            connection.execute("DELETE FROM question_translations_fts")
            connection.execute(
                "INSERT INTO question_translations_fts(question_id, locale, statement_raw) "
                "SELECT question_id, locale, json_extract(payload_json, '$.statement_raw') "
                "FROM question_translations"
            )
            row = connection.execute("SELECT count(*) AS c FROM questions_fts").fetchone()
        return int(row["c"])

    # -- Questions ----------------------------------------------------------
    def question_exists(self, question_id: str) -> bool:
        """Checks whether a question with the given identifier exists."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM questions WHERE id = ? LIMIT 1", (question_id,)
            ).fetchone()
        return row is not None

    def insert_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> str:
        """Inserts a new question and returns the insertion status.

        In the ``local`` environment it computes the ``canonical_hash`` of the
        statement and, if a question with the same hash already exists in the
        same course, blocks the insertion by returning ``"DUPLICATE"``. The
        ``embedding`` (optional) is stored for later semantic deduplication.

        Raises:
            DatabaseError: if a question with the same ``id`` already exists.
        """
        if self.settings.APP_ENV == "local":
            question = question.model_copy(
                update={"canonical_hash": compute_canonical_hash(question.statement_raw)}
            )
        values = self._question_values(question, embedding)
        with self.connect() as connection:
            if self.settings.APP_ENV == "local":
                duplicate = connection.execute(
                    "SELECT 1 FROM questions "
                    "WHERE course = ? AND json_extract(payload_json, '$.canonical_hash') = ? "
                    "LIMIT 1",
                    (question.course, question.canonical_hash),
                ).fetchone()
                if duplicate is not None:
                    return "DUPLICATE"
            try:
                connection.execute(
                    "INSERT INTO questions("
                    "id, course, topic, subtopic, difficulty, prerequisites_json, "
                    "payload_json, created_at, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"a question with id {question.id!r} already exists") from exc
        return "INSERTED"

    def update_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> None:
        """Updates an existing question and synchronizes FTS5.

        If ``embedding`` is ``None``, the stored embedding is kept as is (it is
        not deleted); to empty it explicitly, pass an empty list.

        Raises:
            DatabaseError: if the question does not exist.
        """
        values = self._question_values(question, embedding)
        with self.connect() as connection:
            if embedding is None:
                cursor = connection.execute(
                    "UPDATE questions SET course = ?, topic = ?, subtopic = ?, difficulty = ?, "
                    "prerequisites_json = ?, payload_json = ?, created_at = ? WHERE id = ?",
                    (
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[0],
                    ),
                )
            else:
                cursor = connection.execute(
                    "UPDATE questions SET course = ?, topic = ?, subtopic = ?, difficulty = ?, "
                    "prerequisites_json = ?, payload_json = ?, created_at = ?, embedding_json = ? "
                    "WHERE id = ?",
                    (
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[0],
                    ),
                )
            if cursor.rowcount == 0:
                raise DatabaseError(f"no question exists with id {question.id!r}")

    def upsert_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> None:
        """Inserts or replaces a question while keeping FTS5 coherent."""
        values = self._question_values(question, embedding)
        with self.connect() as connection:
            connection.execute("DELETE FROM questions WHERE id = ?", (question.id,))
            connection.execute(
                "INSERT INTO questions("
                "id, course, topic, subtopic, difficulty, prerequisites_json, "
                "payload_json, created_at, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def get_question(self, question_id: str) -> EnrichedQuestion | None:
        """Retrieves a validated question or ``None`` if it does not exist."""
        from taygeta.core.schema import EnrichedQuestion

        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
        if row is None:
            return None
        return EnrichedQuestion.model_validate_json(row["payload_json"])

    def delete_question(self, question_id: str) -> bool:
        """Deletes a question (and its translations and FTS rows via triggers)."""
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM question_translations WHERE question_id = ?", (question_id,)
            )
            cursor = connection.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        return cursor.rowcount > 0

    def count_questions(self) -> int:
        """Returns the total number of stored questions."""
        with self.connect() as connection:
            row = connection.execute("SELECT count(*) AS c FROM questions").fetchone()
        return int(row["c"])

    def list_question_ids(self) -> list[str]:
        """Returns the identifiers of all questions, ordered by id."""
        with self.connect() as connection:
            rows = connection.execute("SELECT id FROM questions ORDER BY id").fetchall()
        return [row["id"] for row in rows]

    def list_questions(
        self,
        *,
        course: str | None = None,
        topic: str | None = None,
        subtopic: str | None = None,
        difficulty: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EnrichedQuestion]:
        """Lists questions applying curricular filters (parameterized queries)."""
        from taygeta.core.schema import EnrichedQuestion

        conditions: list[str] = []
        params: list[Any] = []
        if course is not None:
            conditions.append("course = ?")
            params.append(course)
        if topic is not None:
            conditions.append("topic = ?")
            params.append(topic)
        if subtopic is not None:
            conditions.append("subtopic = ?")
            params.append(subtopic)
        if difficulty is not None:
            conditions.append("difficulty = ?")
            params.append(difficulty)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            f"SELECT payload_json FROM questions {where} "
            "ORDER BY created_at DESC, id ASC LIMIT ? OFFSET ?"
        )
        params.extend([self._clamp_limit(limit), max(0, offset)])
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [EnrichedQuestion.model_validate_json(row["payload_json"]) for row in rows]

    def search_questions(
        self,
        *,
        query: str | None = None,
        course: str | None = None,
        topic: str | None = None,
        subtopic: str | None = None,
        difficulty: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EnrichedQuestion]:
        """Retrieves questions by text (FTS5) and/or curricular filters.

        Uses parameterized queries and a bounded limit. It never returns
        solutions or correct answers.
        """
        from taygeta.core.schema import EnrichedQuestion

        conditions: list[str] = []
        params: list[Any] = []

        if query:
            source = (
                "FROM questions_fts JOIN questions q ON q.id = questions_fts.id "
                "WHERE questions_fts MATCH ?"
            )
            params.append(_build_match_query(query))
            order = "ORDER BY bm25(questions_fts), q.created_at DESC"
        else:
            source = "FROM questions q WHERE 1 = 1"
            order = "ORDER BY q.created_at DESC, q.id ASC"

        if course is not None:
            conditions.append("q.course = ?")
            params.append(course)
        if topic is not None:
            conditions.append("q.topic = ?")
            params.append(topic)
        if subtopic is not None:
            conditions.append("q.subtopic = ?")
            params.append(subtopic)
        if difficulty is not None:
            conditions.append("q.difficulty = ?")
            params.append(difficulty)

        extra = f" AND {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT q.payload_json {source}{extra} {order} LIMIT ? OFFSET ?"
        params.extend([self._clamp_limit(limit), max(0, offset)])

        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [EnrichedQuestion.model_validate_json(row["payload_json"]) for row in rows]

    # -- Translations -------------------------------------------------------
    def upsert_question_translation(self, translation: QuestionTranslation) -> None:
        """Inserts or updates the translated copy of a question for a locale."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO question_translations"
                "(question_id, locale, payload_json, source, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(question_id, locale) DO UPDATE SET "
                "payload_json = excluded.payload_json, source = excluded.source, "
                "created_at = excluded.created_at",
                (
                    translation.question_id,
                    translation.locale,
                    translation.model_dump_json(),
                    translation.source,
                    translation.created_at,
                ),
            )

    def get_question_translation(self, question_id: str, locale: str) -> QuestionTranslation | None:
        """Retrieves the translated copy of a question, or ``None`` if it does not exist."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM question_translations "
                "WHERE question_id = ? AND locale = ?",
                (question_id, locale),
            ).fetchone()
        if row is None:
            return None
        return QuestionTranslation.model_validate_json(row["payload_json"])

    def list_question_translations(self, question_id: str) -> list[QuestionTranslation]:
        """Lists all stored translated copies of a question."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM question_translations "
                "WHERE question_id = ? ORDER BY locale",
                (question_id,),
            ).fetchall()
        return [QuestionTranslation.model_validate_json(row["payload_json"]) for row in rows]

    def delete_question_translations(self, question_id: str) -> int:
        """Deletes all translated copies of a question. Returns how many it deleted."""
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM question_translations WHERE question_id = ?", (question_id,)
            )
        return cursor.rowcount

    def search_translation_ids(
        self, query: str, *, locale: str | None = None, limit: int = 20
    ) -> list[str]:
        """Ids of questions whose translated statement matches ``query`` (FTS5)."""
        conditions = ["question_translations_fts MATCH ?"]
        params: list[Any] = [_build_match_query(query)]
        if locale is not None:
            conditions.append("locale = ?")
            params.append(locale)
        sql = (
            "SELECT DISTINCT question_id FROM question_translations_fts "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY bm25(question_translations_fts) LIMIT ?"
        )
        params.append(self._clamp_limit(limit))
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [row["question_id"] for row in rows]

    # -- Taxonomy -----------------------------------------------------------
    def upsert_course(self, slug: str) -> None:
        """Registers a course in the catalog (the names live in ``taxonomy_labels``)."""
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO courses(slug) VALUES (?)", (slug,))

    def get_course_name(self, slug: str, *, locale: str = DEFAULT_LOCALE) -> str | None:
        """Returns the localized name of a course, or ``None`` if it is not labeled."""
        return self.get_taxonomy_label("course", slug, locale=locale)

    def get_taxonomy_label(
        self,
        kind: str,
        slug: str,
        *,
        parent_slug: str = "",
        locale: str = DEFAULT_LOCALE,
    ) -> str | None:
        """Returns the taxonomy label for ``locale``, or ``None`` if it does not exist."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT label FROM taxonomy_labels "
                "WHERE kind = ? AND parent_slug = ? AND slug = ? AND locale = ?",
                (kind, parent_slug, slug, locale),
            ).fetchone()
        return row["label"] if row else None

    def upsert_taxonomy_label(
        self,
        kind: str,
        slug: str,
        label: str,
        *,
        locale: str,
        parent_slug: str = "",
        source: str = "manual",
    ) -> None:
        """Inserts a label without overwriting a higher-precedence source (manual > seed > llm)."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO taxonomy_labels"
                "(kind, parent_slug, slug, locale, label, source) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(kind, parent_slug, slug, locale) DO UPDATE SET "
                "label = excluded.label, source = excluded.source "
                f"WHERE {_SOURCE_RANK_SQL}",
                (kind, parent_slug, slug, locale, label, source),
            )

    def taxonomy_label_map(self, *, locale: str) -> dict[tuple[str, str, str], str]:
        """``(kind, parent_slug, slug) -> label`` map for ``locale`` (for caches)."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT kind, parent_slug, slug, label FROM taxonomy_labels WHERE locale = ?",
                (locale,),
            ).fetchall()
        return {(row["kind"], row["parent_slug"], row["slug"]): row["label"] for row in rows}

    def upsert_taxonomy_names(self, names: Iterable[TaxonomyNames], *, source: str = "llm") -> int:
        """Persists localized names (es/en) generated by the LLM.

        Writes each locale as an independent label respecting precedence
        (manual > seed > llm), so a generated name never overwrites a seeded or
        hand-edited one. Returns how many rows were attempted to be written.
        """
        written = 0
        for entry in names:
            for locale, label in entry.locale_labels():
                self.upsert_taxonomy_label(
                    entry.kind,
                    entry.slug,
                    label,
                    locale=locale,
                    parent_slug=entry.parent_slug,
                    source=source,
                )
                written += 1
        return written

    def upsert_topic(
        self,
        course_slug: str,
        topic_slug: str,
        subtopic_slug: str | None = None,
    ) -> int:
        """Inserts if missing and returns the topic/subtopic identifier."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM topics WHERE course_slug = ? AND topic_slug = ? "
                "AND COALESCE(subtopic_slug, '') = COALESCE(?, '')",
                (course_slug, topic_slug, subtopic_slug),
            ).fetchone()
            if row is not None:
                return int(row["id"])
            try:
                cursor = connection.execute(
                    "INSERT INTO topics(course_slug, topic_slug, subtopic_slug) VALUES (?, ?, ?)",
                    (course_slug, topic_slug, subtopic_slug),
                )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(
                    f"could not create topic {topic_slug!r}: course {course_slug!r} does not exist"
                ) from exc
            return int(cursor.lastrowid or 0)

    def upsert_prerequisite(self, slug: str, description: str | None = None) -> None:
        """Inserts or updates a prerequisite of the catalog."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO prerequisites(slug, description) VALUES (?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET description = COALESCE(excluded.description, "
                "prerequisites.description)",
                (slug, description),
            )

    def add_prerequisites(self, slugs: list[str]) -> list[str]:
        """Dynamically adds new prerequisite tags to the catalog."""
        slugs = list(dict.fromkeys(slug for slug in slugs if slug))
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO prerequisites(slug, description) VALUES (?, NULL) "
                "ON CONFLICT(slug) DO NOTHING",
                [(slug,) for slug in slugs],
            )
        return slugs

    def list_courses(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT slug FROM courses ORDER BY slug").fetchall()
        return [dict(row) for row in rows]

    def list_topics(self, course_slug: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if course_slug is None:
                rows = connection.execute(
                    "SELECT id, course_slug, topic_slug, subtopic_slug FROM topics "
                    "ORDER BY course_slug, topic_slug, subtopic_slug"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, course_slug, topic_slug, subtopic_slug FROM topics "
                    "WHERE course_slug = ? ORDER BY topic_slug, subtopic_slug",
                    (course_slug,),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_prerequisites(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT slug, description FROM prerequisites ORDER BY slug"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_taxonomy(self, *, locale: str | None = None) -> dict[str, Any]:
        """Taxonomy catalog, with localized names.

        By default each entry includes a ``names`` map (``{locale: label}``) with
        all supported locales; topics and subtopics are resolved in the context of
        their course (``parent_slug``). With ``locale`` a flat, clean shape is
        returned with a single ``name`` per entry. Labels come from
        ``taxonomy_labels`` and fall back to ``title_case_slug``.
        """
        labels = self._label_map()
        with self.connect() as connection:
            courses = [
                dict(row)
                for row in connection.execute("SELECT slug FROM courses ORDER BY slug").fetchall()
            ]
            topics = [
                dict(row)
                for row in connection.execute(
                    "SELECT id, course_slug, topic_slug, subtopic_slug FROM topics "
                    "ORDER BY course_slug, topic_slug, subtopic_slug"
                ).fetchall()
            ]
            prerequisites = [
                dict(row)
                for row in connection.execute(
                    "SELECT slug, description FROM prerequisites ORDER BY slug"
                ).fetchall()
            ]
        for course in courses:
            course["names"] = self._localized_names(labels, "course", course["slug"])
        for topic in topics:
            parent = topic["course_slug"]
            topic["names"] = self._localized_names(
                labels, "topic", topic["topic_slug"], parent_slug=parent
            )
            subtopic = topic.get("subtopic_slug")
            if subtopic:
                topic["subtopic_names"] = self._localized_names(
                    labels, "subtopic", subtopic, parent_slug=parent
                )
        for prerequisite in prerequisites:
            description = prerequisite.get("description")
            names: dict[str, str] = {}
            for locale_key in SUPPORTED_LOCALES:
                label = labels.get(("prerequisite", "", prerequisite["slug"], locale_key))
                if not label and locale_key == DEFAULT_LOCALE and description:
                    label = description
                names[locale_key] = label or title_case_slug(
                    prerequisite["slug"], locale=locale_key
                )
            prerequisite["names"] = names
        active = locale if is_supported_locale(locale) else None
        if active is None:
            return {"courses": courses, "topics": topics, "prerequisites": prerequisites}
        return {
            "locale": active,
            "courses": [
                {"slug": course["slug"], "name": course["names"][active]} for course in courses
            ],
            "topics": [
                {
                    "id": topic["id"],
                    "course_slug": topic["course_slug"],
                    "topic_slug": topic["topic_slug"],
                    "subtopic_slug": topic.get("subtopic_slug"),
                    "name": topic["names"][active],
                    **(
                        {"subtopic_name": topic["subtopic_names"][active]}
                        if topic.get("subtopic_slug")
                        else {}
                    ),
                }
                for topic in topics
            ],
            "prerequisites": [
                {"slug": prerequisite["slug"], "name": prerequisite["names"][active]}
                for prerequisite in prerequisites
            ],
        }

    def _label_map(self) -> dict[tuple[str, str, str, str], str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT kind, parent_slug, slug, locale, label FROM taxonomy_labels"
            ).fetchall()
        return {
            (row["kind"], row["parent_slug"], row["slug"], row["locale"]): row["label"]
            for row in rows
        }

    @staticmethod
    def _localized_names(
        labels: dict[tuple[str, str, str, str], str],
        kind: str,
        slug: str,
        *,
        parent_slug: str = "",
    ) -> dict[str, str]:
        return {
            locale: labels.get((kind, parent_slug, slug, locale))
            or title_case_slug(slug, locale=locale)
            for locale in SUPPORTED_LOCALES
        }

    def get_live_taxonomy_tree(self) -> dict[str, dict[str, list[str]]]:
        """Returns the live curricular taxonomy of the database.

        Rebuilds the ``course -> topic -> subtopics`` hierarchy actually in use
        from the already-confirmed questions. Only courses and topics present in
        the ``questions`` table are included, so the LLM can reuse existing slugs
        instead of inventing variants. The dictionary does not store solutions or
        correct answers: only curricular slugs.
        """
        tree: dict[str, dict[str, list[str]]] = {}
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT course, topic, subtopic FROM questions "
                "WHERE course IS NOT NULL ORDER BY course, topic, subtopic"
            ).fetchall()
        for row in rows:
            course = row["course"]
            topic = row["topic"]
            subtopic = row["subtopic"]
            topics = tree.setdefault(course, {})
            subtopics = topics.setdefault(topic, [])
            if subtopic and subtopic not in subtopics:
                subtopics.append(subtopic)
        return tree

    # -- Semantic deduplication ---------------------------------------------
    def get_embedding(self, question_id: str) -> list[float] | None:
        """Returns the stored embedding of a question, or ``None`` if it has none."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT embedding_json FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
        if row is None or not row["embedding_json"]:
            return None
        return decode_embedding(row["embedding_json"])

    def list_embeddings(self, *, course: str | None = None) -> list[tuple[str, list[float]]]:
        """Lists ``(id, embedding)`` of the questions that already have a vector.

        It is used to compute cosine similarity in memory (the current volume is
        low). If ``course`` is given, it is restricted to that course.
        """
        if course is not None:
            sql = (
                "SELECT id, embedding_json FROM questions "
                "WHERE course = ? AND embedding_json IS NOT NULL ORDER BY id"
            )
            params: tuple[Any, ...] = (course,)
        else:
            sql = (
                "SELECT id, embedding_json FROM questions "
                "WHERE embedding_json IS NOT NULL ORDER BY id"
            )
            params = ()
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        result: list[tuple[str, list[float]]] = []
        for row in rows:
            vector = decode_embedding(row["embedding_json"])
            if vector:
                result.append((row["id"], vector))
        return result

    def find_similar_questions(
        self,
        embedding: Sequence[float],
        *,
        threshold: float,
        course: str | None = None,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """Returns the questions with cosine similarity >= ``threshold``.

        Sorts by descending similarity and bounds to ``limit`` results. It
        compares only against questions that already have a stored embedding.
        """
        matches: list[tuple[str, float]] = []
        for question_id, vector in self.list_embeddings(course=course):
            similarity = cosine_similarity(embedding, vector)
            if similarity >= threshold:
                matches.append((question_id, similarity))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    # -- Staging ------------------------------------------------------------
    def create_staging_session(
        self,
        data: dict[str, Any] | list[Any] | str,
        *,
        status: str = "NEEDS_ATTENTION",
        session_id: str | None = None,
    ) -> str:
        """Creates a staging session and returns its identifier."""
        self._validate_status(status)
        sid = session_id or uuid.uuid4().hex
        data_json = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO staging_sessions(session_id, data_json, created_at, status) "
                "VALUES (?, ?, ?, ?)",
                (sid, data_json, utc_now_iso(), status),
            )
        return sid

    def get_staging_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT session_id, data_json, created_at, status FROM staging_sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        session = dict(row)
        session["data"] = json.loads(session.pop("data_json"))
        return session

    def update_staging_session(
        self,
        session_id: str,
        *,
        data: dict[str, Any] | list[Any] | str | None = None,
        status: str | None = None,
    ) -> bool:
        """Updates the data and/or the status of a staging session."""
        if data is None and status is None:
            raise DatabaseError("no change was given for the staging session")
        if status is not None:
            self._validate_status(status)

        assignments: list[str] = []
        params: list[Any] = []
        if data is not None:
            assignments.append("data_json = ?")
            params.append(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        params.append(session_id)

        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE staging_sessions SET {', '.join(assignments)} WHERE session_id = ?",
                params,
            )
        return cursor.rowcount > 0

    def delete_staging_session(self, session_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM staging_sessions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    def list_staging_sessions(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT session_id, created_at, status FROM staging_sessions "
                    "ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT session_id, created_at, status FROM staging_sessions "
                    "WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
        return [dict(row) for row in rows]

    # -- State --------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Returns a verifiable summary of the database state."""
        with self.connect() as connection:
            tables = [
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            counts: dict[str, int] = {}
            for table in ("questions", "courses", "topics", "prerequisites", "staging_sessions"):
                counts[table] = int(
                    connection.execute(f"SELECT count(*) AS c FROM {table}").fetchone()["c"]
                )
            version_row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "db_path": self.db_path,
            "schema_version": version_row["value"] if version_row else None,
            "fts5_enabled": self._fts5_enabled,
            "integrity": integrity,
            "tables": tables,
            "counts": counts,
        }

    # -- Internal utilities -------------------------------------------------
    @staticmethod
    def _clamp_limit(limit: int) -> int:
        if limit < 1:
            return 1
        return min(limit, MAX_SEARCH_LIMIT)

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in STAGING_STATUSES:
            raise DatabaseError(f"invalid staging status {status!r}; use one of {STAGING_STATUSES}")

    @staticmethod
    def _question_values(
        question: EnrichedQuestion, embedding: Sequence[float] | None = None
    ) -> tuple[Any, ...]:
        return (
            question.id,
            question.course,
            question.topic,
            question.subtopic,
            question.difficulty,
            json.dumps(question.prerequisites, ensure_ascii=False),
            question.model_dump_json(),
            question.created_at,
            encode_embedding(embedding),
        )


def get_database() -> Database:
    """Returns a :class:`Database` instance over the effective path.

    Uses ``DATABASE_URL`` if configured; otherwise ``DB_PATH``.
    """
    return Database(get_settings().database_path)
