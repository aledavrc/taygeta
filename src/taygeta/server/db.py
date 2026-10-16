"""Persistencia SQLite con búsqueda textual FTS5 para Taygeta.

Este módulo implementa la base de datos descrita en la sección 6 de la
especificación:

* ``questions`` con los campos exactos solicitados y ``payload_json`` con el
  objeto :class:`~taygeta.core.schema.EnrichedQuestion` serializado.
* ``questions_fts`` (virtual FTS5) sincronizada mediante triggers AFTER
  INSERT/UPDATE/DELETE, de modo que la búsqueda textual nunca queda desfasada.
* ``courses``, ``topics``, ``prerequisites`` y ``staging_sessions``.

La base de datos vive en ``DB_PATH`` (por defecto ``/app/data/taygeta.db``), es
decir, en el volumen persistente, nunca en la raíz del repositorio. La
inicialización del esquema es idempotente y :meth:`Database.status` informa del
estado real de la base de datos. No se almacenan soluciones ni respuestas
correctas: el ``payload_json`` está validado por esquemas que las prohíben.
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
from taygeta.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, title_case_slug
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

#: Versión lógica del esquema, registrada en ``schema_meta``.
SCHEMA_VERSION = "3"

#: Estados admitidos para una sesión de staging.
STAGING_STATUSES: tuple[str, ...] = ("VALID", "NEEDS_ATTENTION", "CONFIRMED", "DISCARDED")

#: Límite máximo de resultados para proteger el servicio RAG.
MAX_SEARCH_LIMIT = 100

#: Fuentes de un label de taxonomía, de menor a mayor precedencia al escribir.
TAXONOMY_LABEL_SOURCES: tuple[str, ...] = ("llm", "seed", "manual")

#: Precedencia numérica usada por el ``ON CONFLICT`` de ``taxonomy_labels``.
_SOURCE_RANK_SQL = (
    "CASE excluded.source WHEN 'manual' THEN 3 WHEN 'seed' THEN 2 ELSE 1 END >= "
    "CASE taxonomy_labels.source WHEN 'manual' THEN 3 WHEN 'seed' THEN 2 ELSE 1 END"
)


class DatabaseError(RuntimeError):
    """Error controlado de acceso o integridad de la base de datos."""


#: DDL de las tablas de taxonomía. Se reconstruyen al migrar desde el esquema
#: anterior, en el que los nombres de curso vivían en ``courses.name``.
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

-- Nombres legibles por idioma y tipo de taxonomía. Los slugs siguen siendo
-- datos estables; ``source`` ordena la precedencia manual > seed > llm.
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

-- El subtema puede ser NULL; COALESCE evita que varios NULL se consideren distintos.
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

-- Copias traducidas por idioma. El enunciado canónico vive en ``questions``;
-- estas filas son derivadas, no entran en la deduplicación y se borran en
-- cascada con su pregunta.
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

-- Sincronización FTS5 con questions mediante triggers.
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

-- Sincronización FTS5 de las traducciones mediante triggers.
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
    """Convierte una consulta libre en una expresión MATCH segura para FTS5.

    Cada término se entrecomilla para evitar que la sintaxis de FTS5 (operadores
    ``AND``/``OR``/``NEAR``, comillas, paréntesis, ``*``) provoque errores o
    consultas inesperadas.
    """
    tokens = query.split()
    if not tokens:
        raise DatabaseError("la consulta de búsqueda no puede estar vacía")
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class Database:
    """Acceso transaccional a la base de datos SQLite de Taygeta."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        initialize: bool = True,
    ) -> None:
        """Crea el acceso a la base de datos.

        Args:
            db_path: Ruta del archivo SQLite. Si se omite, se usa ``DB_PATH``
                de la configuración.
            settings: Configuración a usar; por defecto la global.
            initialize: Si es ``True``, inicializa el esquema de forma
                idempotente al construir la instancia.
        """
        self.settings = settings or get_settings()
        resolved = db_path if db_path is not None else self.settings.database_path
        self.db_path = str(resolved)
        self._memory_connection: sqlite3.Connection | None = None
        self._fts5_enabled = False
        if initialize:
            self.initialize()

    # -- Conexión -----------------------------------------------------------
    @property
    def is_memory(self) -> bool:
        """Indica si la base de datos reside en memoria."""
        return self.db_path == ":memory:"

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Contexto transaccional que confirma al salir y revierte ante error."""
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

    # -- Inicialización -----------------------------------------------------
    def initialize(self) -> None:
        """Crea el esquema de forma idempotente (seguro de llamar varias veces).

        Raises:
            DatabaseError: si la ruta no es accesible o el esquema no puede crearse.
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
                f"no se pudo inicializar la base de datos en {self.db_path!r}: {exc}"
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
        """Aplica migraciones incrementales a bases de datos ya existentes.

        ``CREATE TABLE IF NOT EXISTS`` no altera tablas preexistentes, por lo que
        las columnas añadidas en versiones posteriores del esquema se crean aquí
        de forma idempotente (por ejemplo ``questions.embedding_json``).

        Los nombres de curso dejaron de vivir en ``courses.name`` (ahora viven en
        ``taxonomy_labels``); si se detecta el esquema anterior, las tablas de
        taxonomía se reconstruyen desde cero. Los datos almacenados son
        desechables, de modo que no se migran filas.
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
        """Siembra ``taxonomy_labels`` la primera vez (idempotente, no sobrescribe)."""
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
        """Regenera las tablas FTS5 y devuelve el número de filas de preguntas."""
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

    # -- Preguntas ----------------------------------------------------------
    def question_exists(self, question_id: str) -> bool:
        """Comprueba si existe una pregunta con el identificador indicado."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM questions WHERE id = ? LIMIT 1", (question_id,)
            ).fetchone()
        return row is not None

    def insert_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> str:
        """Inserta una pregunta nueva y devuelve el estado de la inserción.

        En entorno ``local`` calcula el ``canonical_hash`` del enunciado y, si ya
        existe una pregunta con el mismo hash en el mismo curso, bloquea la
        inserción devolviendo ``"DUPLICATE"``. El ``embedding`` (opcional) se
        almacena para la deduplicación semántica posterior.

        Raises:
            DatabaseError: si ya existe una pregunta con el mismo ``id``.
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
                raise DatabaseError(f"ya existe una pregunta con id {question.id!r}") from exc
        return "INSERTED"

    def update_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> None:
        """Actualiza una pregunta existente y sincroniza FTS5.

        Si ``embedding`` es ``None``, el embedding almacenado se conserva tal cual
        (no se borra); para vaciarlo explícitamente pásese una lista vacía.

        Raises:
            DatabaseError: si la pregunta no existe.
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
                raise DatabaseError(f"no existe una pregunta con id {question.id!r}")

    def upsert_question(
        self, question: EnrichedQuestion, *, embedding: Sequence[float] | None = None
    ) -> None:
        """Inserta o reemplaza una pregunta manteniendo FTS5 coherente."""
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
        """Recupera una pregunta validada o ``None`` si no existe."""
        from taygeta.core.schema import EnrichedQuestion

        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
        if row is None:
            return None
        return EnrichedQuestion.model_validate_json(row["payload_json"])

    def delete_question(self, question_id: str) -> bool:
        """Elimina una pregunta (y sus traducciones y filas FTS vía triggers)."""
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM question_translations WHERE question_id = ?", (question_id,)
            )
            cursor = connection.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        return cursor.rowcount > 0

    def count_questions(self) -> int:
        """Devuelve el número total de preguntas almacenadas."""
        with self.connect() as connection:
            row = connection.execute("SELECT count(*) AS c FROM questions").fetchone()
        return int(row["c"])

    def list_question_ids(self) -> list[str]:
        """Devuelve los identificadores de todas las preguntas, ordenados por id."""
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
        """Lista preguntas aplicando filtros curriculares (consultas parametrizadas)."""
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
        """Recupera preguntas por texto (FTS5) y/o filtros curriculares.

        Usa consultas parametrizadas y un límite acotado. Nunca devuelve
        soluciones ni respuestas correctas.
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

    # -- Traducciones -------------------------------------------------------
    def upsert_question_translation(self, translation: QuestionTranslation) -> None:
        """Inserta o actualiza la copia traducida de una pregunta para un idioma."""
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
        """Recupera la copia traducida de una pregunta, o ``None`` si no existe."""
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
        """Lista todas las copias traducidas almacenadas de una pregunta."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM question_translations "
                "WHERE question_id = ? ORDER BY locale",
                (question_id,),
            ).fetchall()
        return [QuestionTranslation.model_validate_json(row["payload_json"]) for row in rows]

    def delete_question_translations(self, question_id: str) -> int:
        """Elimina todas las copias traducidas de una pregunta. Devuelve cuántas borró."""
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM question_translations WHERE question_id = ?", (question_id,)
            )
        return cursor.rowcount

    def search_translation_ids(
        self, query: str, *, locale: str | None = None, limit: int = 20
    ) -> list[str]:
        """Ids de preguntas cuyo enunciado traducido coincide con ``query`` (FTS5)."""
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

    # -- Taxonomía ----------------------------------------------------------
    def upsert_course(self, slug: str) -> None:
        """Registra un curso en el catálogo (los nombres viven en ``taxonomy_labels``)."""
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO courses(slug) VALUES (?)", (slug,))

    def get_course_name(self, slug: str, *, locale: str = DEFAULT_LOCALE) -> str | None:
        """Devuelve el nombre localizado de un curso, o ``None`` si no está etiquetado."""
        return self.get_taxonomy_label("course", slug, locale=locale)

    def get_taxonomy_label(
        self,
        kind: str,
        slug: str,
        *,
        parent_slug: str = "",
        locale: str = DEFAULT_LOCALE,
    ) -> str | None:
        """Devuelve el label de taxonomía para ``locale``, o ``None`` si no existe."""
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
        """Inserta un label sin pisar una fuente de mayor precedencia (manual > seed > llm)."""
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
        """Mapa ``(kind, parent_slug, slug) -> label`` para ``locale`` (para cachés)."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT kind, parent_slug, slug, label FROM taxonomy_labels WHERE locale = ?",
                (locale,),
            ).fetchall()
        return {(row["kind"], row["parent_slug"], row["slug"]): row["label"] for row in rows}

    def upsert_taxonomy_names(self, names: Iterable[TaxonomyNames], *, source: str = "llm") -> int:
        """Persiste nombres localizados (es/en) generados por el LLM.

        Escribe cada idioma como un label independiente respetando la precedencia
        (manual > seed > llm), de modo que un nombre generado nunca pisa uno
        sembrado o editado a mano. Devuelve cuántas filas se intentaron escribir.
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
        """Inserta si falta y devuelve el identificador del tema/subtema."""
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
                    f"no se pudo crear el tema {topic_slug!r}: el curso {course_slug!r} no existe"
                ) from exc
            return int(cursor.lastrowid or 0)

    def upsert_prerequisite(self, slug: str, description: str | None = None) -> None:
        """Inserta o actualiza un prerrequisito del catálogo."""
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO prerequisites(slug, description) VALUES (?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET description = COALESCE(excluded.description, "
                "prerequisites.description)",
                (slug, description),
            )

    def add_prerequisites(self, slugs: list[str]) -> list[str]:
        """Añade dinámicamente nuevos tags de prerrequisitos al catálogo."""
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

    def list_taxonomy(self) -> dict[str, Any]:
        """Catálogo de taxonomía con nombres localizados por idioma soportado.

        Cada entrada incluye un mapa ``names`` (``{locale: label}``); los temas y
        subtemas se resuelven en el contexto de su curso (``parent_slug``). Los
        labels proceden de ``taxonomy_labels`` y caen a ``title_case_slug``.
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
            for locale in SUPPORTED_LOCALES:
                label = labels.get(("prerequisite", "", prerequisite["slug"], locale))
                if not label and locale == DEFAULT_LOCALE and description:
                    label = description
                names[locale] = label or title_case_slug(prerequisite["slug"], locale=locale)
            prerequisite["names"] = names
        return {"courses": courses, "topics": topics, "prerequisites": prerequisites}

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
        """Devuelve la taxonomía curricular viva de la base de datos.

        Reconstruye la jerarquía ``curso -> tema -> subtemas`` realmente en uso a
        partir de las preguntas ya confirmadas. Solo se incluyen cursos y temas
        presentes en la tabla ``questions``, de modo que el LLM pueda reutilizar
        los slugs existentes en lugar de inventar variantes. El diccionario no
        almacena soluciones ni respuestas correctas: únicamente slugs curriculares.
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

    # -- Deduplicación semántica -------------------------------------------
    def get_embedding(self, question_id: str) -> list[float] | None:
        """Devuelve el embedding almacenado de una pregunta, o ``None`` si no tiene."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT embedding_json FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
        if row is None or not row["embedding_json"]:
            return None
        return decode_embedding(row["embedding_json"])

    def list_embeddings(self, *, course: str | None = None) -> list[tuple[str, list[float]]]:
        """Lista ``(id, embedding)`` de las preguntas que ya tienen vector.

        Se usa para calcular la similitud coseno en memoria (el volumen actual es
        bajo). Si se indica ``course``, se restringe a ese curso.
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
        """Devuelve las preguntas con similitud coseno >= ``threshold``.

        Ordena por similitud descendente y acota a ``limit`` resultados. Compara
        solo contra preguntas que ya tienen embedding almacenado.
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
        """Crea una sesión de staging y devuelve su identificador."""
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
        """Actualiza los datos y/o el estado de una sesión de staging."""
        if data is None and status is None:
            raise DatabaseError("no se indicó ningún cambio para la sesión de staging")
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

    # -- Estado -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Devuelve un resumen verificable del estado de la base de datos."""
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

    # -- Utilidades internas ------------------------------------------------
    @staticmethod
    def _clamp_limit(limit: int) -> int:
        if limit < 1:
            return 1
        return min(limit, MAX_SEARCH_LIMIT)

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in STAGING_STATUSES:
            raise DatabaseError(
                f"estado de staging inválido {status!r}; use uno de {STAGING_STATUSES}"
            )

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
    """Devuelve una instancia de :class:`Database` sobre la ruta efectiva.

    Usa ``DATABASE_URL`` si está configurada; en caso contrario ``DB_PATH``.
    """
    return Database(get_settings().database_path)
