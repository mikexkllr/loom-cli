"""The character set Loom draws with.

Two sets: box-drawing/Unicode for real terminals, ASCII for the ones that
would otherwise print question marks (legacy Windows code pages, ``LOOM_ASCII=1``).
Every renderer asks for glyphs by *name* so the fallback is total — there is no
surface that only looks right in UTF-8.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Glyphs:
    # rails — the weave: one vertical thread per active agent
    rail: str  # a thread running down the gutter
    rail_dim: str  # a delegated (nested) thread
    branch: str  # a tool call hanging off the thread
    corner: str  # the thread ending
    result: str  # a tool result under its call

    # box drawing
    h: str  # horizontal rule
    tl: str
    tr: str
    bl: str
    br: str
    v: str

    # marks
    logo: str  # the weave motif in the wordmark
    dot: str  # separator between metadata fields
    local: str  # runs here, free
    cloud: str  # runs remotely, billed
    ok: str
    fail: str
    warn: str
    pending: str
    think: str
    prompt: str  # the input caret
    arrow: str  # "delegates to"
    bullet: str
    tip: str  # an aside worth reading, distinct from a thread bullet
    stop: str
    loop: str
    plus: str
    minus: str
    ellipsis: str

    spinner: tuple[str, ...]  # the shuttle crossing the warp


UNICODE = Glyphs(
    rail="│",
    rail_dim="╎",
    branch="├",
    corner="╰",
    result="⤷",
    h="─",
    tl="╭",
    tr="╮",
    bl="╰",
    br="╯",
    v="│",
    logo="▞▚▞",
    dot="·",
    local="⌂",
    cloud="☁",
    ok="✔",
    fail="✘",
    warn="▲",
    pending="◇",
    think="✻",
    prompt="❯",
    arrow="→",
    bullet="◆",
    tip="▸",
    stop="■",
    loop="↻",
    plus="+",
    minus="-",
    ellipsis="…",
    # Braille frames read as a shuttle travelling across the warp and back.
    spinner=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
)

ASCII = Glyphs(
    rail="|",
    rail_dim=":",
    branch="+",
    corner="`",
    result=">",
    h="-",
    tl="+",
    tr="+",
    bl="+",
    br="+",
    v="|",
    logo="//\\",
    dot="-",
    local="[L]",
    cloud="[C]",
    ok="OK",
    fail="X",
    warn="!",
    pending="?",
    think="*",
    prompt=">",
    arrow="->",
    bullet="*",
    tip=">",
    stop="#",
    loop="~",
    plus="+",
    minus="-",
    ellipsis="...",
    spinner=("-", "\\", "|", "/"),
)


def glyphs(unicode: bool = True) -> Glyphs:
    return UNICODE if unicode else ASCII
