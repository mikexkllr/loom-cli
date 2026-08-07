"""The wordmark and welcome card.

The mark is "LOOM" in half-blocks with the two O's woven — ``▚▞`` is a twill
crossing, so the logo is literally a scrap of fabric. It fades from the warp
colour to the cloud colour across its width, which is the same warm→cool axis
the rest of the UI uses to mean local→cloud.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.text import Text

from loom.ui import render
from loom.ui.render import ink
from loom.ui.theme import theme_of

WORDMARK = (
    r"█    ▄▀▀▄ ▄▀▀▄ █▄  ▄█",
    r"█    █▚▞█ █▚▞█ █ ▀▀ █",
    r"▀▀▀▀  ▀▀   ▀▀  ▀    ▀",
)

ASCII_WORDMARK = (
    r" _    ___   ___  __  __ ",
    r"| |  / _ \ / _ \|  \/  |",
    r"|__| \___/ \___/|_|\/|_|",
)


def blend(a: str, b: str, t: float) -> str:
    """Blend two ``#rrggbb`` colours. Shared with :mod:`loom.ui.about`, which
    runs the same warm→cool axis across a whole screen."""
    ai = (int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16))
    bi = (int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ai, bi))


def wordmark(console: Console) -> Text:
    theme = theme_of(console)
    lines = WORDMARK if theme.unicode else ASCII_WORDMARK
    out = Text()
    if theme.mono:
        for i, line in enumerate(lines):
            out.append(line + ("\n" if i < len(lines) - 1 else ""), style="loom.bright")
        return out
    palette = theme.palette
    span = max(len(line) for line in lines) - 1 or 1
    for i, line in enumerate(lines):
        for x, char in enumerate(line):
            out.append(char, style=blend(palette.warp, palette.cloud, x / span))
        if i < len(lines) - 1:
            out.append("\n")
    return out


def welcome(console: Console, *, version: str, roles, cwd: str, tagline: str = "") -> Group:
    """The card Loom opens a session with: who you are talking to, what is
    running where, and where you are."""
    g = ink(console)
    head = Text()
    head.append_text(wordmark(console))
    body = [head, Text()]
    line = Text("  ")
    line.append(f"v{version}", style="loom.muted")
    if tagline:
        line.append_text(render.dot(console))
        line.append(tagline, style="loom.muted")
    body.append(line)
    if roles:
        body.append(Text())
        body.append(render.fleet_table(console, roles, header=False))
    body.append(Text())
    body.append(Text(f"{g.prompt} ", style="loom.line") + Text(cwd, style="loom.muted"))
    return Group(*body)


def hint_line(console: Console, items) -> Text:
    """The quiet strip under the welcome card: ``/help · /status · shift+tab``."""
    out = Text("  ")
    out.append_text(render.join(console, [Text(i, style="loom.muted") for i in items]))
    return out
