"""Rich theme + console derived from ui settings."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

from loom.core.settings import UISettings

_THEMES = {
    "dark": {
        "loom.user": "bold cyan",
        "loom.agent": "white",
        "loom.tool": "magenta",
        "loom.subagent": "green",
        "loom.dim": "grey50",
        "loom.warn": "yellow",
        "loom.err": "bold red",
        "loom.accent": "bold blue",
    },
    "light": {
        "loom.user": "bold blue",
        "loom.agent": "black",
        "loom.tool": "purple",
        "loom.subagent": "dark_green",
        "loom.dim": "grey42",
        "loom.warn": "dark_orange",
        "loom.err": "bold red",
        "loom.accent": "bold blue",
    },
    "mono": {
        "loom.user": "bold",
        "loom.agent": "default",
        "loom.tool": "dim",
        "loom.subagent": "bold",
        "loom.dim": "dim",
        "loom.warn": "bold",
        "loom.err": "bold",
        "loom.accent": "bold",
    },
}


def make_console(ui: UISettings) -> Console:
    theme_name = ui.theme
    if theme_name == "auto":
        theme_name = "dark"  # Rich adapts to terminal; dark palette is a safe default
    styles = _THEMES.get(theme_name, _THEMES["dark"])
    return Console(theme=Theme(styles))
