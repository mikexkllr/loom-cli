"""``/about`` — Loom weaving itself, then the facts.

The animation is a loom seen head on. Bare warp threads hang the full height
of the terminal; a shuttle runs back and forth laying weft, and the cloth
grows down the screen behind it. The wordmark isn't drawn on top of the
cloth — it is *in* it, revealed row by row as the weave reaches it, because
the twill glyphs the logo is made of (``▚▞``) are the same ones the cloth is
made of.

It runs on the same warm→cool axis as the rest of the UI: threads start at
the local colour on the left and end at the cloud colour on the right, so
even the splash screen says what Loom is about.

Everything is degradable. No terminal (a pipe, CI, a captured buffer) means
no animation at all — just the card. No UTF-8 means ASCII thread glyphs. No
colour (``mono``, ``NO_COLOR``) means the structure carries it alone. And
Ctrl+C at any point skips straight to the card, which is the part with the
information in it.
"""

from __future__ import annotations

import os
import time

from rich.console import Console
from rich.text import Text

from loom.ui import banner as banner_mod
from loom.ui import render
from loom.ui.render import ink
from loom.ui.theme import theme_of

# One full crossing of the shuttle every this many frames. Fast enough to
# read as machinery, slow enough that the eye can follow the shuttle.
FRAMES_PER_PASS = 7
DURATION = 2.0  # seconds of weaving
FPS = 24
HOLD = 0.45  # seconds on the finished cloth before the card
MIN_WIDTH = 24
MIN_HEIGHT = 10

TAGLINE = "hybrid local/cloud agent fleet"


def _toward_bg(colour: str, amount: float, dark: bool) -> str:
    """Push a colour towards the terminal background — the cloth has to sit
    behind the wordmark, and dimming is what puts it there. Light themes have
    to go *up* towards white, not down towards black."""
    return banner_mod.blend(colour, "#0F1117" if dark else "#FFFFFF", amount)


class _Palette:
    """Per-cell styles for one console, resolved once per animation.

    Colours are quantised into bands across the width so a row is a handful
    of styled runs instead of one run per column.
    """

    BANDS = 24

    def __init__(self, console: Console, width: int) -> None:
        theme = theme_of(console)
        self.mono = theme.mono
        self.width = max(width, 1)
        if self.mono:
            return
        p = theme.palette
        span = self.BANDS - 1 or 1
        self.warp = [_toward_bg(banner_mod.blend(p.local, p.cloud, i / span), 0.72, p.dark) for i in range(self.BANDS)]
        self.cloth = [_toward_bg(banner_mod.blend(p.local, p.cloud, i / span), 0.45, p.dark) for i in range(self.BANDS)]
        self.weft = [banner_mod.blend(p.local, p.cloud, i / span) for i in range(self.BANDS)]
        self.mark = [f"bold {banner_mod.blend(p.warp, p.cloud, i / span)}" for i in range(self.BANDS)]
        self.shuttle_style = f"bold {p.warp}"

    def style(self, kind: str, x: int) -> str:
        if self.mono:
            return {
                "warp": "loom.line",
                "cloth": "loom.muted",
                "weft": "loom.text",
                "shuttle": "loom.bright",
                "mark": "loom.bright",
                "tagline": "loom.muted",
                "clear": "loom.line",
            }[kind]
        if kind == "shuttle":
            return self.shuttle_style
        band = min(self.BANDS - 1, x * self.BANDS // self.width)
        if kind in ("tagline", "clear"):
            return self.cloth[band]
        return {"warp": self.warp, "cloth": self.cloth, "weft": self.weft, "mark": self.mark}[kind][band]


def shuttle_at(frame: int, width: int) -> int:
    """Where the shuttle is on its current pass — a triangle wave, so it
    reverses at each selvedge like the real thing."""
    span = max(width - 1, 1)
    pass_index, step = divmod(frame, FRAMES_PER_PASS)
    position = step * span // max(FRAMES_PER_PASS - 1, 1)
    return position if pass_index % 2 == 0 else span - position


PLAQUE_PAD = (2, 1)  # columns, rows of clear cloth around the mark


def _overlay(width: int, height: int, lines: tuple[str, ...], version: str) -> dict[tuple[int, int], tuple[str, str]]:
    """``(row, col) -> (char, kind)`` for the centred wordmark and the line
    under it, plus the clear patch they sit in.

    The patch matters: the twill is the same two glyphs the logo is drawn
    with, so without it the cloth fills the counters of the letters and the
    mark disappears into its own texture. Returned as a lookup so the frame
    builder stays one flat loop.
    """
    art_width = max(len(line) for line in lines)
    caption = f"v{version}  {TAGLINE}"
    cap_width = len(caption) if len(caption) + 4 <= width else 0
    block_width = max(art_width, cap_width)
    block_height = len(lines) + (2 if cap_width else 0)
    top = max((height - block_height) // 2, 0)
    left = max((width - block_width) // 2, 0)
    pad_x, pad_y = PLAQUE_PAD

    cells: dict[tuple[int, int], tuple[str, str]] = {}
    for y in range(max(top - pad_y, 0), min(top + block_height + pad_y, height)):
        for x in range(max(left - pad_x, 0), min(left + block_width + pad_x, width)):
            cells[(y, x)] = (" ", "clear")
    art_left = left + (block_width - art_width) // 2
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char != " ":
                cells[(top + y, art_left + x)] = (char, "mark")
    if cap_width:
        cap_row = top + len(lines) + 1
        cap_left = left + (block_width - cap_width) // 2
        for x, char in enumerate(caption):
            if char != " " and cap_row < height:
                cells[(cap_row, cap_left + x)] = (char, "tagline")
    return cells


def frame(
    console: Console,
    *,
    width: int,
    height: int,
    woven: int,
    shuttle: int,
    forward: bool,
    version: str,
    palette: _Palette | None = None,
) -> Text:
    """One still of the loom.

    ``woven`` is how many rows of cloth exist; the row at that index is the
    one on the shuttle right now, and everything below it is still bare warp.
    Pure: given the same arguments it draws the same screen, which is what
    makes the animation testable without a terminal.
    """
    theme = theme_of(console)
    g = ink(console)
    unicode = theme.unicode
    palette = palette or _Palette(console, width)
    lines = banner_mod.WORDMARK if unicode else banner_mod.ASCII_WORDMARK
    overlay = _overlay(width, height, lines, version)

    twill = "▚▞" if unicode else "\\/"
    warp_chars = (g.rail, g.rail_dim)
    weft_char = g.h

    out = Text(no_wrap=True, overflow="crop")
    for y in range(height):
        # Coalesce equal styles into runs — a full screen of per-cell appends
        # every frame is the difference between an animation and a slideshow.
        run: list[str] = []
        run_style = ""
        for x in range(width):
            if y < woven:
                char, kind = twill[(x + y) % 2], "cloth"
            elif y == woven:
                laid = x <= shuttle if forward else x >= shuttle
                if x == shuttle:
                    char, kind = g.bullet, "shuttle"
                elif laid:
                    char, kind = weft_char, "weft"
                else:
                    char, kind = warp_chars[x % 2], "warp"
            else:
                char, kind = warp_chars[x % 2], "warp"
            # The mark is woven in, not printed over: it appears only on cloth
            # that is finished, so it is revealed row by row as the weave
            # passes it. Never on the shuttle's own row — the weft is still
            # being laid there.
            if y < woven and (y, x) in overlay:
                char, kind = overlay[(y, x)]
            style = palette.style(kind, x)
            if style != run_style:
                if run:
                    out.append("".join(run), style=run_style)
                run, run_style = [], style
            run.append(char)
        if run:
            out.append("".join(run), style=run_style)
        if y < height - 1:
            out.append("\n")
    return out


def frames(console: Console, *, width: int, height: int, version: str, count: int):
    """The whole animation, as renderables. Ends on finished cloth."""
    palette = _Palette(console, width)
    steps = max(count - 1, 1)
    for i in range(count):
        woven = min(int(i / steps * height), height)
        shuttle = shuttle_at(i, width)
        forward = (i // FRAMES_PER_PASS) % 2 == 0
        yield frame(
            console,
            width=width,
            height=height,
            woven=woven,
            shuttle=shuttle,
            forward=forward,
            version=version,
            palette=palette,
        )


def can_animate(console: Console) -> bool:
    """Animation is for a person watching a terminal. A pipe, a CI log or a
    captured test buffer gets thousands of escape codes and no benefit, and
    ``LOOM_NO_ANIM=1`` is the opt-out for anyone who just wants the facts."""
    if not console.is_terminal or os.environ.get("LOOM_NO_ANIM"):
        return False
    width, height = console.size
    return width >= MIN_WIDTH and height >= MIN_HEIGHT


def animate(console: Console, *, version: str, duration: float = DURATION, fps: int = FPS) -> bool:
    """Weave the screen. Returns True if it actually played.

    Runs in the alternate screen buffer, so the transcript underneath is
    untouched — the REPL scrollback is exactly where it was when the card
    prints. Never raises: a splash screen that breaks ``/about`` would be a
    poor trade.
    """
    if not can_animate(console):
        return False
    width, height = console.size
    count = max(int(duration * fps), FRAMES_PER_PASS * 2)
    delay = duration / count
    try:
        from rich.live import Live

        # `Live(screen=True)`, not `console.screen()`: the bare screen context
        # prints each frame where the cursor happens to be, so the frames
        # scroll past each other instead of repainting in place. Live is what
        # homes the cursor between them.
        with Live(console=console, screen=True, auto_refresh=False, transient=False) as live:
            for still in frames(console, width=width, height=height, version=version, count=count):
                live.update(still, refresh=True)
                time.sleep(delay)
            time.sleep(HOLD)
    except KeyboardInterrupt:
        return True  # skipped on purpose — the card still follows
    except Exception:
        return False
    return True


def card(console: Console, *, cwd: str | None = None):
    """Who you're talking to, what build it is, and where it came from."""
    from loom import __version__
    from loom.core import config as cfg
    from loom.core import uninstall as un
    from loom.core import update as update_mod

    g = ink(console)
    frozen = update_mod.is_frozen()
    if frozen:
        try:
            build = f"standalone binary ({update_mod.asset_name()})"
        except RuntimeError:
            build = "standalone binary"
    else:
        import sys

        build = f"source install {g.dot} python {sys.version.split()[0]}"

    rows = [
        ("what", Text("a fleet, not a single model", style="loom.text")),
        (
            "",
            Text(
                "a cloud orchestrator plans and routes; local subagents do the bounded work "
                "in their own context windows and return only summaries",
                style="loom.muted",
            ),
        ),
        ("build", Text(build, style="loom.text")),
    ]
    binaries = un.installed_binaries()
    if binaries:
        rows.append(("binary", Text(str(binaries[0]), style="loom.muted")))
    rows.append(("home", Text(str(cfg.USER_CONFIG_DIR), style="loom.muted")))
    if cwd:
        rows.append(("project", Text(str(cwd), style="loom.muted")))
    rows += [
        ("repo", Text(f"https://github.com/{update_mod.REPO}", style="loom.warp")),
        ("licence", Text("MIT", style="loom.muted")),
        ("woven from", Text("deepagents · LangGraph · Rich · Typer · prompt_toolkit", style="loom.muted")),
    ]

    head = Text()
    head.append_text(banner_mod.wordmark(console))
    title = Text("  ")
    title.append(f"v{__version__}", style="loom.muted")
    title.append_text(render.dot(console))
    title.append(TAGLINE, style="loom.muted")
    body = render.stack(head, Text(), title, Text(), render.kv(rows, justify="right"))
    return render.card(console, body, title="about")


def show(console: Console, *, cwd: str | None = None, still: bool = False) -> None:
    """The whole ``/about``: the weave, then the card."""
    from loom import __version__

    if not still:
        animate(console, version=__version__)
    console.print()
    console.print(card(console, cwd=cwd))
    render.note(
        console,
        f"[loom.warp]loom update[/loom.warp] fetches the latest build {ink(console).dot} "
        f"[loom.warp]loom uninstall[/loom.warp] removes it {ink(console).dot} "
        f"[loom.warp]/about still[/loom.warp] skips the weave",
        kind="tip",
    )
