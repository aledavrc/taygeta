"""Human-in-the-Loop review UI in NiceGUI.

Event-driven state model (no reruns): the page is built once per session and
events mutate Python objects and refresh the containers.
"""

from __future__ import annotations

from nicegui import ui

__all__ = ["caption"]


def caption(text: str) -> None:
    """Equivalent of ``st.caption``: a small, gray label."""
    ui.label(text).classes("text-xs text-gray-500")
