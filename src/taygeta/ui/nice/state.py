"""NiceGUI UI session state (isolated per browser client).

NiceGUI has no rerun model: the page is built once per session and events
mutate Python objects that refresh the containers. This module keeps a
per-``client.id`` registry so a page swap (e.g. after reloading the URL) does
not lose the staging in progress; the states are released when the client
disconnects (see ``app.on_disconnect`` in ``app.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from taygeta.ingest.pipeline import StagingResult


@dataclass
class QuestionState:
    """Review state of a single question (edited figures, views…).

    ``figures`` is the source of truth during review: it reflects the
    deletions, re-crops and additions the user makes over the figures
    structured by the pipeline.
    """

    figures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionState:
    """Live state of a review session (per browser client)."""

    processed_path: str | None = None
    staging: StagingResult | None = None
    questions: dict[int, QuestionState] = field(default_factory=dict)
    #: Assets of orphan figures (deleted or without placeholder) pending purge.
    orphans: dict[str, set[str]] = field(default_factory=dict)
    processing: bool = False


_sessions: dict[str, SessionState] = {}


def register(client_id: str) -> SessionState:
    """Return (creating it if missing) the session state of the client."""
    return _sessions.setdefault(client_id, SessionState())


def drop(client_id: str) -> None:
    """Release a session's state when its client disconnects."""
    _sessions.pop(client_id, None)
