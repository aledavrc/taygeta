"""Interface localization wiring for the NiceGUI HITL UI.

Bridges the framework-agnostic :mod:`taygeta.i18n` layer with NiceGUI:

* resolves the active locale from ``app.storage.user`` (persisted across
  reloads, like the dark-mode preference);
* renders the floating language switcher;
* loads the matching Quasar language pack so built-in widget strings (selects,
  dialogs, …) follow the chosen locale instead of the startup default.

Importing this module installs the locale resolver, so callers only need to
render the switcher (``language_switcher``) and activate the language pack
(``install_ui_language``) while building a page.
"""

from __future__ import annotations

from nicegui import app, ui

from taygeta.i18n import (
    DEFAULT_LOCALE,
    LOCALE_LABELS,
    SUPPORTED_LOCALES,
    set_locale_resolver,
    t,
)

__all__ = [
    "active_locale",
    "install_ui_language",
    "language_switcher",
    "quasar_locale",
    "set_active_locale",
]

#: Maps a Taygeta locale to the Quasar language pack that NiceGUI ships.
_QUASAR_LOCALES: dict[str, str] = {
    "es": "es",
    "en": "en-US",
}


def quasar_locale(locale: str) -> str:
    """Return the Quasar language pack name for a Taygeta ``locale``."""
    return _QUASAR_LOCALES.get(locale, _QUASAR_LOCALES[DEFAULT_LOCALE])


def active_locale() -> str | None:
    """Return the locale stored for the current session, if any.

    Returns ``None`` when there is no client context (e.g. at import time);
    :func:`taygeta.i18n.get_locale` turns that into :data:`DEFAULT_LOCALE`.
    """
    try:
        return app.storage.user.get("locale")
    except Exception:  # noqa: BLE001 - no active client/storage in this context
        return None


def set_active_locale(locale: str) -> None:
    """Persist ``locale`` for the session and reload the page."""
    if locale not in SUPPORTED_LOCALES:
        return
    app.storage.user["locale"] = locale
    ui.navigate.reload()


def language_switcher() -> None:
    """Render the floating language chooser (next to the dark-mode toggle)."""
    current = active_locale() or DEFAULT_LOCALE
    button = ui.button(icon="translate").props("round flat color=primary")
    button.classes("fixed top-4 right-20 z-50")
    button.mark("language-switcher")
    button.tooltip(t("ui.language.tooltip"))
    with button:
        with ui.menu():
            for locale in SUPPORTED_LOCALES:
                item = ui.menu_item(
                    LOCALE_LABELS[locale],
                    on_click=lambda loc=locale: set_active_locale(loc),
                )
                if locale == current:
                    item.props("active")


def install_ui_language(locale: str) -> None:
    """Load and activate the Quasar language pack matching ``locale``.

    NiceGUI ships the language pack for its startup ``language`` only, so when
    the session uses a different locale the pack is fetched on demand from the
    already-served ``/static/lang/`` directory. Any failure is swallowed: our
    own strings (through :func:`taygeta.i18n.t`) are always correct, so a
    missing pack only leaves Quasar's built-in widget text at the default.
    """
    ui.add_head_html(_quasar_language_script(quasar_locale(locale)))


def _quasar_language_script(quasar_language: str) -> str:
    key = quasar_language.replace("-", "")
    return f"""
<script>
(function () {{
  var ISO = "{quasar_language}";
  var KEY = "{key}";
  function apply() {{
    try {{
      if (window.Quasar && Quasar.Lang && Quasar.Lang[KEY] && Quasar.lang && Quasar.lang.set) {{
        Quasar.lang.set(Quasar.Lang[KEY]);
      }}
    }} catch (err) {{ console.error("Quasar language:", err); }}
  }}
  function install() {{
    if (window.Quasar && Quasar.Lang && Quasar.Lang[KEY]) {{ apply(); return; }}
    var base = null;
    document.querySelectorAll('script[src*="/static/lang/"]').forEach(function (el) {{
      var match = el.src.match(/^(.*\\/static\\/lang\\/)/);
      if (match && !base) base = match[1];
    }});
    if (!base) return;
    var tag = document.createElement("script");
    tag.src = base + ISO + ".umd.prod.js";
    tag.onload = apply;
    document.head.appendChild(tag);
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", install);
  }} else {{
    install();
  }}
}})();
</script>
"""


# Install the resolver as soon as the UI localization layer is imported.
set_locale_resolver(active_locale)
