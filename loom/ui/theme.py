"""Loom's palette.

The colours carry meaning, not just decoration. Two axes run through every
surface:

* **warm = local = free.** Amber/ochre — the colour of raw thread — marks
  anything running on this machine at no cost.
* **cool = cloud = billed.** Sky/indigo marks anything that costs money.

So a glance at a transcript tells you where the work happened and what it
cost, before you read a word of it. Everything else (rules, borders, dim
prose) stays low-contrast so those two signals are the only loud things on
screen.

Each theme is a :class:`Palette` of hex colours; Rich styles are derived from
it so a new theme only has to name ~12 colours. Terminals that can't do
truecolor get the nearest 256/16-colour match from Rich automatically, and
``mono`` drops colour entirely for pipes, CI logs and e-ink terminals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rich.console import Console
from rich.theme import Theme

from loom.core.settings import UISettings


@dataclass(frozen=True)
class Palette:
    """The twelve colours a Loom theme has to name."""

    name: str
    dark: bool
    # structure
    line: str  # borders, rules, separators
    muted: str  # de-emphasised prose, metadata
    text: str  # body text
    bright: str  # headings, emphasis
    # signal
    warp: str  # Loom's signature accent — the primary thread
    local: str  # warm: runs here, costs nothing
    cloud: str  # cool: runs remotely, costs money
    tool: str  # tool calls
    good: str  # success, added lines
    warn: str  # warnings, pending approval
    bad: str  # errors, removed lines
    think: str  # reasoning traces
    # The rail colours cycled across concurrent subagent threads, so two
    # subagents streaming at once are never the same colour.
    threads: tuple[str, ...] = ()

    def thread_color(self, index: int) -> str:
        return self.threads[index % len(self.threads)] if self.threads else self.warp


LOOM_DARK = Palette(
    name="loom",
    dark=True,
    line="#545C70",
    muted="#7C8599",
    text="#C3CAD9",
    bright="#EEF1F7",
    warp="#7DD3C0",
    local="#E8B24A",
    cloud="#6EA8FE",
    tool="#C792EA",
    good="#9ECE6A",
    warn="#E0AF68",
    bad="#F07178",
    think="#646D82",
    threads=("#7DD3C0", "#C792EA", "#6EA8FE", "#E8B24A", "#9ECE6A", "#F78C6C", "#89DDFF"),
)

LOOM_LIGHT = Palette(
    name="loom-light",
    dark=False,
    line="#AAB3C5",
    muted="#6E7688",
    text="#2E3440",
    bright="#12151C",
    warp="#0E8F78",
    local="#9A6B00",
    cloud="#2C5FD0",
    tool="#8250DF",
    good="#3B7D28",
    warn="#9A6B00",
    bad="#C4314B",
    think="#8A92A3",
    threads=("#0E8F78", "#8250DF", "#2C5FD0", "#9A6B00", "#3B7D28", "#B44B22", "#0F7B8F"),
)

# A CRT in a basement. Entirely serious, entirely green.
PHOSPHOR = Palette(
    name="phosphor",
    dark=True,
    line="#317045",
    muted="#4CA366",
    text="#8FE39A",
    bright="#D6FFDC",
    warp="#4AF08A",
    local="#B8F04A",
    cloud="#4AE0D0",
    tool="#7CF0B8",
    good="#4AF08A",
    warn="#D8F04A",
    bad="#F0704A",
    think="#2E6B3F",
    threads=("#4AF08A", "#7CF0B8", "#4AE0D0", "#B8F04A", "#8FE39A", "#3FC96F", "#5FFFC0"),
)

PALETTES: dict[str, Palette] = {p.name: p for p in (LOOM_DARK, LOOM_LIGHT, PHOSPHOR)}

# Older configs (and habit) say dark/light/auto.
ALIASES = {"dark": "loom", "light": "loom-light", "auto": "loom", "default": "loom"}

THEME_NAMES = (*PALETTES, "mono")


def resolve_palette(name: str) -> Palette:
    return PALETTES.get(ALIASES.get(name, name), LOOM_DARK)


def styles_for(name: str) -> dict[str, str]:
    """Rich style table for a theme name. ``mono`` is derived structurally
    (bold/dim only) so Loom stays legible with NO_COLOR, in a pipe, or on a
    terminal that renders colour as mud."""
    if ALIASES.get(name, name) == "mono":
        styles = {
            "loom.line": "dim",
            "loom.muted": "dim",
            "loom.text": "default",
            "loom.bright": "bold",
            "loom.warp": "bold",
            "loom.local": "default",
            "loom.cloud": "bold",
            "loom.tool": "default",
            "loom.good": "bold",
            "loom.warn": "bold",
            "loom.bad": "bold reverse",
            "loom.think": "dim italic",
            "loom.rule": "dim",
            "loom.key": "dim",
            "loom.badge": "reverse",
            "loom.badge.warp": "reverse bold",
            "loom.badge.warn": "reverse bold",
            "loom.badge.bad": "reverse bold",
            "prompt": "default",
            "prompt.choices": "bold",
            "prompt.default": "dim",
            "prompt.invalid": "bold reverse",
            "prompt.invalid.choice": "bold reverse",
            # legacy aliases (see below)
            "loom.user": "bold",
            "loom.agent": "default",
            "loom.subagent": "bold",
            "loom.dim": "dim",
            "loom.err": "bold reverse",
            "loom.accent": "bold",
        }
        # Every ``.b`` and per-thread name the renderers reach for has to exist
        # in every theme, or Rich raises on a style it can't parse. Without
        # colour there is nothing to vary, so the threads all collapse to plain.
        for key in list(styles):
            styles.setdefault(f"{key}.b", "bold")
        for i in range(len(LOOM_DARK.threads)):
            styles[f"loom.t{i}"] = "default"
            styles[f"loom.t{i}.b"] = "bold"
        return styles

    p = resolve_palette(name)
    styles = {
        "loom.line": p.line,
        "loom.muted": p.muted,
        "loom.text": p.text,
        "loom.bright": f"bold {p.bright}",
        "loom.warp": p.warp,
        "loom.local": p.local,
        "loom.cloud": p.cloud,
        "loom.tool": p.tool,
        "loom.good": p.good,
        "loom.warn": p.warn,
        "loom.bad": p.bad,
        "loom.think": f"italic {p.think}",
        "loom.rule": p.line,
        "loom.key": p.muted,
        # Loud variants, for the few places that need to stop the eye.
        "loom.warp.b": f"bold {p.warp}",
        "loom.bad.b": f"bold {p.bad}",
        "loom.good.b": f"bold {p.good}",
        "loom.warn.b": f"bold {p.warn}",
        # Inverse segments for the status line.
        "loom.badge": f"bold {p.bright} on {p.line}",
        "loom.badge.warp": f"bold {'#0F1117' if p.dark else '#FFFFFF'} on {p.warp}",
        "loom.badge.warn": f"bold {'#0F1117' if p.dark else '#FFFFFF'} on {p.warn}",
        "loom.badge.bad": f"bold {'#0F1117' if p.dark else '#FFFFFF'} on {p.bad}",
    }
    # Rich's own Prompt/Confirm styles. Left alone they render the choice list
    # in stock magenta and the default in stock cyan — the one surface that
    # would still look like generic Rich rather than like Loom.
    styles.update(
        {
            "prompt": p.text,
            "prompt.choices": p.warp,
            "prompt.default": p.muted,
            "prompt.invalid": p.bad,
            "prompt.invalid.choice": p.bad,
        }
    )
    # Names the pre-rebuild code used. Kept so third-party hooks, skills and
    # any user markup written against the old names still render.
    styles.update(
        {
            "loom.user": f"bold {p.warp}",
            "loom.agent": p.text,
            "loom.subagent": p.good,
            "loom.dim": p.muted,
            "loom.err": f"bold {p.bad}",
            "loom.accent": f"bold {p.warp}",
        }
    )
    # Per-thread rail styles: loom.t0 … loom.tN.
    for i, colour in enumerate(p.threads):
        styles[f"loom.t{i}"] = colour
        styles[f"loom.t{i}.b"] = f"bold {colour}"
    # Renderers freely ask for the bold variant of any semantic name; Rich
    # raises on one it can't parse, so make sure they all exist.
    for key in list(styles):
        if not key.endswith(".b"):
            styles.setdefault(f"{key}.b", f"bold {styles[key]}")
    return styles


@dataclass
class ConsoleTheme:
    """What the renderers need to know about the active look."""

    palette: Palette
    mono: bool = False
    unicode: bool = True
    width_hint: int = 0
    _cache: dict = field(default_factory=dict, repr=False)


def active_theme_name(ui: UISettings) -> str:
    """The theme actually in force, honouring NO_COLOR and dumb terminals."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return "mono"
    name = ui.theme
    if ALIASES.get(name, name) not in THEME_NAMES:
        return "loom"
    return ALIASES.get(name, name)


def make_console(ui: UISettings) -> Console:
    """The single Console every Loom surface writes through."""
    name = active_theme_name(ui)
    console = Console(theme=Theme(styles_for(name)), soft_wrap=False)
    console._loom_theme = ConsoleTheme(  # type: ignore[attr-defined]
        palette=resolve_palette(name),
        mono=name == "mono",
        unicode=_supports_unicode(console),
    )
    return console


def theme_of(console: Console) -> ConsoleTheme:
    """The ConsoleTheme attached to a console.

    A Console built anywhere but :func:`make_console` — a test, a hook, a
    third-party embedder — has none of Loom's style names, and Rich raises
    ``MissingStyle`` on the first one it meets. Rather than make every caller
    remember, adopt such a console on sight: push the default styles onto it
    and attach the theme record."""
    theme = getattr(console, "_loom_theme", None)
    if theme is None:
        console.push_theme(Theme(styles_for("loom")))
        theme = ConsoleTheme(palette=LOOM_DARK, unicode=_supports_unicode(console))
        console._loom_theme = theme  # type: ignore[attr-defined]
    return theme


def _supports_unicode(console: Console) -> bool:
    """Box-drawing and block glyphs need a UTF-8 stdout. Windows consoles in a
    legacy code page mangle them into question marks, so fall back to ASCII
    rather than printing rubble."""
    if os.environ.get("LOOM_ASCII"):
        return False
    encoding = (getattr(console.file, "encoding", None) or "").lower()
    if not encoding:
        return True  # capture buffers and pipes without an encoding: assume UTF-8
    return "utf" in encoding
