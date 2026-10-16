#!/usr/bin/env python3
"""One-off migration: cleanup of norm bars ``\\|`` in ``statement_raw``.

The OCR/LLM models sometimes emit ``\\|`` (LaTeX norm bar) instead of ``||``
(Typst norm). The transpiler rejects the ``\\|`` command and the exam
compilation fails. This script walks the questions in ``questions`` and
rewrites ``\\|`` → ``||`` in ``payload_json.statement_raw`` and in each
``payload_json.choices[].content_raw``.

The ``questions_fts`` table is kept in sync automatically: the ``questions_au``
trigger re-extracts ``statement_raw`` from ``payload_json`` when the row is
updated.

Usage:

    uv run python scripts/migrate_norm_bars.py --db /path/to/taygeta.db
    uv run python scripts/migrate_norm_bars.py --db /path/to/taygeta.db --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from taygeta.core.schema import sanitize_norm_bars


def _resolve_db_path(explicit: str | None) -> str:
    """Resolve the database path: explicit argument or configuration."""
    if explicit:
        return explicit
    try:
        from taygeta.core.settings import get_settings

        return get_settings().database_path
    except Exception:  # pragma: no cover - no configuration available
        return "/app/data/taygeta.db"


def migrate(db_path: str, *, dry_run: bool = False) -> int:
    """Rewrite ``\\|`` → ``||`` in the stored statements and choices.

    Returns the number of modified rows.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT id, payload_json FROM questions").fetchall()
        changed = 0
        for row in rows:
            payload = json.loads(row["payload_json"])
            before = json.dumps(payload, ensure_ascii=False, sort_keys=True)

            statement = payload.get("statement_raw")
            if isinstance(statement, str):
                payload["statement_raw"] = sanitize_norm_bars(statement)
            for choice in payload.get("choices") or []:
                content = choice.get("content_raw") if isinstance(choice, dict) else None
                if isinstance(content, str):
                    choice["content_raw"] = sanitize_norm_bars(content)

            after = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if after == before:
                continue
            changed += 1
            if not dry_run:
                connection.execute(
                    "UPDATE questions SET payload_json = ? WHERE id = ?",
                    (json.dumps(payload, ensure_ascii=False), row["id"]),
                )
        if not dry_run:
            connection.commit()
        return changed
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean norm bars \\| in statement_raw.")
    parser.add_argument("--db", default=None, help="Path to the SQLite file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only reports the rows that would be modified, without writing.",
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    try:
        changed = migrate(db_path, dry_run=args.dry_run)
    except sqlite3.Error as exc:
        print(f"error accessing the database {db_path!r}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    action = "would be modified" if args.dry_run else "modified"
    print(f"{changed} question(s) {action} in {db_path}")


if __name__ == "__main__":
    main()
