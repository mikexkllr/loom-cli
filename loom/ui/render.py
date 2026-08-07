"""Loom's drawing primitives.

Everything Loom puts on screen goes through here, so the look is defined in
one place instead of being re-invented per slash command. Two groups:

* **Static** — cards, rules, key/value grids, tables, diffs, choosers. Thin
  borders, left-aligned titles, one accent; the restraint is the point.
* **The weave** (:class:`Weave`) — the live transcript. A multi-agent run is
  drawn as literal threads: the orchestrator holds a rail down the gutter,
  every delegated subagent opens its own indented rail beside it, and the rail
  is coloured warm when the work is local/free and cool when it is billed. The
  shape of the delegation is visible without reading a word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from loom.ui.glyphs import Glyphs, glyphs
from loom.ui.theme import theme_of

# Thin, quiet frames. Rich's default HEAVY_HEAD table is a spreadsheet; Loom
# wants a printed page.
CARD_BOX = box.ROUNDED
TABLE_BOX = box.Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)


def ink(console: Console) -> Glyphs:
    return glyphs(theme_of(console).unicode)


# ---------------------------------------------------------------------------
# Badges — the local/cloud axis, everywhere it appears
# ---------------------------------------------------------------------------


def where(console: Console, is_local: bool, *, label: bool = False) -> Text:
    """``⌂`` warm for local/free, ``☁`` cool for cloud/billed."""
    g = ink(console)
    mark = g.local if is_local else g.cloud
    style = "loom.local" if is_local else "loom.cloud"
    text = Text(mark, style=style)
    if label:
        text.append(" local" if is_local else " cloud", style=style)
    return text


def model_badge(console: Console, model: str, is_local: bool) -> Text:
    out = where(console, is_local)
    out.append(f" {model}", style="loom.local" if is_local else "loom.cloud")
    return out


def dot(console: Console) -> Text:
    return Text(f" {ink(console).dot} ", style="loom.line")


def join(console: Console, parts, style: str = "loom.muted") -> Text:
    """Metadata fields separated by the mid dot, e.g. ``a · b · c``."""
    out = Text()
    for i, part in enumerate(parts):
        if i:
            out.append_text(dot(console))
        out.append_text(part if isinstance(part, Text) else Text(str(part), style=style))
    return out


# ---------------------------------------------------------------------------
# Static blocks
# ---------------------------------------------------------------------------


def card(
    console: Console,
    body: RenderableType,
    *,
    title: str | Text | None = None,
    subtitle: str | Text | None = None,
    style: str = "loom.line",
    pad: tuple[int, int] = (0, 1),
    expand: bool = False,
) -> Panel:
    """A thin frame with its title set into the top rule, left-aligned."""
    return Panel(
        body,
        title=_title(title),
        subtitle=_title(subtitle, style="loom.muted"),
        title_align="left",
        subtitle_align="right",
        border_style=style,
        box=CARD_BOX,
        padding=pad,
        expand=expand,
    )


def _title(value, style: str = "loom.bright") -> Text | None:
    if value is None:
        return None
    return value if isinstance(value, Text) else Text(str(value), style=style)


def rule(console: Console, title: str = "", style: str = "loom.rule") -> None:
    """A section break. Titled rules carry the accent; bare ones are furniture."""
    characters = ink(console).h
    if title:
        console.print(Rule(Text(title, style="loom.bright"), style=style, align="left", characters=characters))
    else:
        console.print(Rule(style=style, characters=characters))


def kv(rows, *, key_style: str = "loom.key", pad: int = 2, justify: str = "right") -> Table:
    """An aligned key/value grid — Loom's answer to every "show me my setup"
    surface. Values may be plain strings (markup allowed) or Text."""
    grid: Table = Table.grid(padding=(0, pad))
    grid.add_column(justify=justify, style=key_style, no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in rows:
        grid.add_row(key, value)
    return grid


def table(console: Console, *columns, header_style: str = "loom.key") -> Table:
    """A lean table: a single rule under the header, no body grid.

    Columns are ``str`` or ``(name, kwargs)``. The console is taken so the
    table's styles are guaranteed to resolve when it is printed.
    """
    theme_of(console)
    out = Table(
        box=TABLE_BOX,
        header_style=header_style,
        border_style="loom.line",
        padding=(0, 2, 0, 0),
        pad_edge=False,
        show_edge=False,
    )
    for column in columns:
        if isinstance(column, tuple):
            name, kwargs = column
            out.add_column(name, **kwargs)
        else:
            out.add_column(column)
    return out


def note(console: Console, message: str, *, kind: str = "muted") -> None:
    """A one-line aside. ``kind`` is muted | good | warn | bad | tip."""
    g = ink(console)
    marks = {
        "good": (g.ok, "loom.good"),
        "warn": (g.warn, "loom.warn"),
        "bad": (g.fail, "loom.bad"),
        "tip": (g.tip, "loom.warp"),
        "muted": ("", "loom.muted"),
    }
    mark, style = marks.get(kind, marks["muted"])
    line = Text()
    if mark:
        line.append(f"{mark} ", style=style)
    line.append_text(Text.from_markup(message, style="loom.muted" if kind == "muted" else "loom.text"))
    console.print(line)


def diff(console: Console, lines, *, limit: int = 80) -> Text:
    """A unified diff with coloured gutters. Header lines are furniture; the
    ``+``/``-`` marks are the only thing that should catch the eye."""
    out = Text()
    for line in lines[:limit]:
        if line.startswith("+++") or line.startswith("---"):
            out.append(line + "\n", style="loom.muted")
        elif line.startswith("@@"):
            out.append(line + "\n", style="loom.warp")
        elif line.startswith("+"):
            out.append(line + "\n", style="loom.good")
        elif line.startswith("-"):
            out.append(line + "\n", style="loom.bad")
        else:
            out.append(line + "\n", style="loom.muted")
    if len(lines) > limit:
        out.append(f"{ink(console).ellipsis} +{len(lines) - limit} more diff lines\n", style="loom.muted")
    return out


def choices(console: Console, options, *, title: str | None = None) -> None:
    """The numbered chooser used by approvals and plan-mode hand-off.

    ``options`` is a sequence of ``(key, label)`` or ``(key, label, hint)``.
    """
    if title:
        console.print(Text(title, style="loom.bright"))
    for option in options:
        key, label, *rest = option
        line = Text("  ")
        line.append(f"{key}", style="loom.warp.b")
        line.append("  ")
        line.append_text(Text.from_markup(label, style="loom.text"))
        if rest and rest[0]:
            line.append(f"  {rest[0]}", style="loom.muted")
        console.print(line)


def confirm(console: Console, question: str, *, default: bool = True, on_eof: bool = False) -> bool:
    """A yes/no prompt that survives having no one to ask.

    Piped input, CI, and a closed stdin all make :class:`rich.prompt.Confirm`
    raise rather than return, which would crash a headless run on a question
    it could simply have declined. ``on_eof`` is what to assume then, and it
    defaults to *no* — every caller here guards something with a side effect
    (a multi-gigabyte download, an install), and silently doing that because
    nobody answered is the wrong way to be wrong.
    """
    from rich.prompt import Confirm

    try:
        return bool(Confirm.ask(question, default=default, console=console))
    except (EOFError, KeyboardInterrupt, OSError):
        return on_eof


def ask(console: Console, prompt: str, *, options=None, default: str | None = None) -> str:
    """A single-line prompt in Loom's caret style."""
    from rich.prompt import Prompt

    return Prompt.ask(
        Text(f"  {ink(console).prompt}", style="loom.warp") + Text(f" {prompt}", style="loom.text"),
        choices=list(options) if options else None,
        default=default,
        console=console,
        show_choices=bool(options),
    )


# ---------------------------------------------------------------------------
# The weave — live transcript rendering
# ---------------------------------------------------------------------------


@dataclass
class Thread:
    """One agent's thread in the weave."""

    label: str
    model: str = ""
    is_local: bool = True
    depth: int = 0
    slot: int = 0

    @property
    def style(self) -> str:
        return f"loom.t{self.slot}"


_WS = re.compile(r"(\s+)")

# Streaming markdown-lite. Real CommonMark rendering (rich.markdown.Markdown)
# treats every un-fenced newline as a "soft break" and joins it into the
# surrounding paragraph — correct per spec, but it strips the indentation off
# any code the model streams without fencing, which is exactly what Gutter
# exists to preserve. So this stays line-based: recognise structure at the
# start of a line and inline spans within it, but never join two lines the
# model actually sent.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^([-*+])\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\d{1,9}[.)])\s+(.*)$")
# Longest-match-first, so a lone "*" never wins over "**" starting at the same
# spot. No underscore markers (_x_, __x__): snake_case and dunder identifiers
# (`foo_bar`, `__init__`) are everywhere in a coding assistant's output, and
# treating every underscore as a possible emphasis toggle would mangle them.
_INLINE_MARKERS = ("**", "~~", "`", "*")
_MARKER_KIND = {"**": "bold", "~~": "strike", "*": "italic"}
# A span still open after this many words is abandoned rather than styling
# the rest of the line — protects a lone "*args" or "**kwargs" mentioned in
# prose (no closing marker is ever coming) from bolding everything after it.
_SPAN_MAX_AGE = 12


class Gutter:
    """Streams text into a rail-prefixed column, wrapping at word boundaries.

    Tokens arrive a fragment at a time, so this holds back only the trailing
    partial word — enough to never split a word across a wrap, little enough
    that output still appears as it is generated rather than line by line.

    A light markdown pass runs alongside: headings, list markers, fenced
    code, and inline emphasis/code spans all render live, word by word, as
    they arrive — see the module-level note above for why this is line-based
    rather than a real CommonMark parse.
    """

    def __init__(self, console: Console, prefix: Text, *, style: str | None = None) -> None:
        self.console = console
        self.prefix = prefix
        self.style = style
        self._col = 0
        self._pending = ""
        self._fresh = True  # nothing written on the current line yet
        self._wrapped = False  # ...and we got here by wrapping, not a newline
        self._at_line_start = True
        self._in_fence = False
        self._in_code_span = False
        self._heading_active = False
        self._active: dict[str, int] = {}  # span kind -> word index it opened at
        self._word_index = 0

    @property
    def width(self) -> int:
        """Recomputed on every access rather than cached, so a live terminal
        resize takes effect on the next line instead of the next block."""
        return max(20, self.console.width - cell_len(self.prefix.plain) - 1)

    def write(self, text: str) -> None:
        buf = self._pending + text
        self._pending = ""
        cut = max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))
        if cut < 0:
            self._pending = buf
            return
        self._pending = buf[cut + 1 :]
        self._raw(buf[: cut + 1])

    def flush(self) -> None:
        if self._pending:
            self._raw(self._pending)
            self._pending = ""

    def newline(self) -> None:
        self.flush()
        self.console.print()
        self._col = 0
        self._fresh = True
        self._wrapped = False
        self._reset_line_state()

    def blank(self) -> None:
        """An empty rail line — the breathing room between blocks."""
        self.flush()
        if not self._fresh:
            self.console.print()
        self.console.print(self.prefix)
        self._col = 0
        self._fresh = True
        self._wrapped = False
        self._reset_line_state()

    def block(self, renderable: RenderableType) -> None:
        """Render a full Rich renderable (markdown, diff, table) inside the rail."""
        self.flush()
        if not self._fresh:
            self.console.print()
            self._col = 0
            self._fresh = True
        self._wrapped = False
        options = self.console.options.update(width=self.width)
        rendered = self.console.render_lines(renderable, options, pad=False)
        # Renderables that end in a newline (diffs, Markdown) leave a trailing
        # empty line, which would print as a bare rail stub.
        while rendered and not "".join(s.text for s in rendered[-1]).strip():
            rendered.pop()
        for segments in rendered:
            line = self.prefix.copy()
            for segment in segments:
                line.append(segment.text, style=segment.style)
            line.rstrip()  # in place; trailing padding would drag the rail wide
            self.console.print(line)
        self._col = 0
        self._fresh = True
        self._wrapped = False

    # -- internals
    def _raw(self, chunk: str) -> None:
        hide_break = False  # the previous line was a fence delimiter, printed nowhere
        for i, line in enumerate(chunk.split("\n")):
            if i:
                if not hide_break:
                    self.console.print()
                self._col = 0
                self._fresh = True
                self._wrapped = False  # a real newline — keep this line's indent
                self._reset_line_state()
            hide_break = self._line(line)

    def _reset_line_state(self) -> None:
        """A genuine newline starts markdown state fresh — spans never cross a
        hard line break the model sent, which both matches how emphasis is
        meant to work and bounds how far a stray marker can bleed. An open
        fence is the one exception: it spans lines by definition, until its
        closing marker arrives."""
        self._at_line_start = True
        self._heading_active = False
        self._in_code_span = False
        self._active = {}
        self._word_index = 0

    def _line(self, line: str) -> bool:
        """Prints one source-line fragment. Returns True if the fragment was
        a fence delimiter that renders nowhere — the caller then knows not to
        print a blank row for the line break that followed it."""
        # An empty fragment carries no content to make a start-of-line call
        # on — it's an artifact of splitting on "\n", not a real line. Tokens
        # commonly arrive as a lone newline character, which produces exactly
        # this empty fragment; consuming the flag here would mean the next
        # *real* line, seconds later, never gets checked for a fence/heading.
        if self._at_line_start and line:
            self._at_line_start = False
            consumed = self._consume_line_start(line)
            if consumed is None:
                return True
            line = consumed
        for word in _WS.split(line):
            if not word:
                continue
            blank = not word.strip()
            # Drop the space that *caused* a wrap, but never the leading
            # whitespace of a genuine new line — that is source indentation,
            # and eating it reflows every code block the model streams.
            if blank and self._fresh and self._wrapped:
                continue
            if self._in_fence:
                text = Text(word, style=self.console.get_style("loom.tool"))
            elif blank or self._heading_active:
                text = Text(word, style=self._base_style())
            else:
                self._word_index += 1
                self._expire_stale_spans()
                text = self._styled_word(word)
            length = cell_len(text.plain)
            if not blank and self._col and self._col + length > self.width:
                self.console.print()
                self._col = 0
                self._fresh = True
                self._wrapped = True
            if self._fresh:
                self.console.print(self.prefix, end="")
                self._fresh = False
            self.console.print(text, end="", soft_wrap=True)
            self._col += length
        return False

    def _consume_line_start(self, line: str) -> str | None:
        """Recognise a block-level marker at the start of a genuine source
        line and strip it. Returns the remaining text to stream normally, or
        ``None`` if the whole line was consumed (a fence delimiter)."""
        if _FENCE_RE.match(line):
            self._in_fence = not self._in_fence
            self._heading_active = False
            self._in_code_span = False
            self._active = {}
            return None
        if self._in_fence:
            return line
        heading = _HEADING_RE.match(line)
        if heading:
            self._heading_active = True
            return heading.group(2)
        marker = _BULLET_RE.match(line) or _ORDERED_RE.match(line)
        if marker:
            self._emit_bullet(marker.group(1))
            return marker.group(2)
        return line

    def _emit_bullet(self, marker: str) -> None:
        g = ink(self.console)
        label = f"{g.bullet} " if marker in ("-", "*", "+") else f"{marker} "
        if self._fresh:
            self.console.print(self.prefix, end="")
            self._fresh = False
        text = Text(label, style=self.console.get_style("loom.muted"))
        self.console.print(text, end="", soft_wrap=True)
        self._col += cell_len(label)

    def _base_style(self) -> Style:
        if self._heading_active:
            return self.console.get_style("loom.bright") + Style(bold=True)
        if self.style:
            return self.console.get_style(self.style)
        return Style()

    def _expire_stale_spans(self) -> None:
        stale = [kind for kind, opened in self._active.items() if self._word_index - opened > _SPAN_MAX_AGE]
        for kind in stale:
            del self._active[kind]

    def _current_style(self) -> Style:
        style = self._base_style()
        if "bold" in self._active:
            style += Style(bold=True)
        if "italic" in self._active:
            style += Style(italic=True)
        if "strike" in self._active:
            style += Style(strike=True)
        return style

    def _styled_word(self, word: str) -> Text:
        """Strip inline markers (bold/italic/strike/code) as they're found,
        styling the text between them live — without waiting for whatever
        later word eventually closes the span."""
        out = Text()
        i, n = 0, len(word)
        while i < n:
            if self._in_code_span:
                j = word.find("`", i)
                code_style = self.console.get_style("loom.tool")
                if j == -1:
                    out.append(word[i:], style=code_style)
                    return out
                if j > i:
                    out.append(word[i:j], style=code_style)
                self._in_code_span = False
                i = j + 1
                continue
            marker, j = self._find_marker(word, i)
            if marker is None:
                out.append(word[i:], style=self._current_style())
                return out
            if j > i:
                out.append(word[i:j], style=self._current_style())
            if marker == "`":
                self._in_code_span = True
            else:
                kind = _MARKER_KIND[marker]
                if kind in self._active:
                    del self._active[kind]
                else:
                    self._active[kind] = self._word_index
            i = j + len(marker)
        return out

    @staticmethod
    def _find_marker(word: str, start: int) -> tuple[str | None, int]:
        best: tuple[str, int] | None = None
        for marker in _INLINE_MARKERS:
            idx = word.find(marker, start)
            if idx == -1:
                continue
            if best is None or idx < best[1] or (idx == best[1] and len(marker) > len(best[0])):
                best = (marker, idx)
        return best if best else (None, -1)


@dataclass
class Weave:
    """The transcript renderer for one turn.

    Call :meth:`open` when a thread starts talking, :meth:`text` as tokens
    arrive, :meth:`tool_call` / :meth:`tool_result` for structure, and
    :meth:`close` at the end of the turn.

    A thread announces itself — name and model badge — the first time it
    speaks in a turn. After that the rail alone identifies it, so a
    conversation that bounces between the orchestrator and a subagent doesn't
    reprint the same header a dozen times.
    """

    console: Console
    flat: bool = False  # no rails: one bullet per block, for pipes and logs
    # The pre-first-token indicator. Owned here so every drawing path can
    # retire it before it draws — a Live region and a print cannot share the
    # terminal, and forgetting one call corrupts the whole transcript.
    working: "Working | None" = None
    _thread: Thread | None = None
    _kind: str = ""
    _gutter: Gutter | None = None
    _slots: dict[str, int] = field(default_factory=dict)
    _stack: list[Thread] = field(default_factory=list)
    _announced: set[str] = field(default_factory=set)

    # -- thread bookkeeping
    def slot_for(self, label: str) -> int:
        """A stable colour slot per actor, assigned in the order they first
        speak — so the orchestrator is always thread 0 and two subagents
        streaming at once never share a colour."""
        if label not in self._slots:
            palette = theme_of(self.console).palette
            self._slots[label] = len(self._slots) % max(1, len(palette.threads) or 1)
        return self._slots[label]

    def thread(self, label: str, model: str = "", is_local: bool = True, depth: int = 0) -> Thread:
        thread = Thread(label=label, model=model, is_local=is_local, depth=depth, slot=self.slot_for(label))
        while len(self._stack) <= depth:
            self._stack.append(thread)
        self._stack[depth] = thread
        del self._stack[depth + 1 :]  # a shallower thread speaking ends the deeper ones
        return thread

    def _settle(self) -> None:
        if self.working is not None:
            self.working.stop()

    def reset(self) -> None:
        """Between turns: colours and headers start fresh."""
        self._thread = None
        self._kind = ""
        self._gutter = None
        self._slots.clear()
        self._stack.clear()
        self._announced.clear()

    # -- prefixes
    def _ancestors(self, thread: Thread) -> Text:
        """The parent threads' rails, so a subagent's output stays visually
        inside the delegation that spawned it."""
        g = ink(self.console)
        out = Text()
        if self.flat:
            return out
        for depth in range(thread.depth):
            parent = self._stack[depth] if depth < len(self._stack) else None
            out.append(f"{g.rail_dim} ", style=parent.style if parent else "loom.line")
        return out

    def _prefix(self, thread: Thread, mark: str) -> Text:
        if self.flat:
            return Text("  ")
        out = self._ancestors(thread)
        out.append(f"{mark} ", style=thread.style)
        return out

    # -- lifecycle
    def open(self, thread: Thread, kind: str = "text") -> None:
        """Start (or switch to) a thread's block."""
        if self._thread and self._thread.label == thread.label and self._kind == kind:
            return
        new_speaker = not self._thread or self._thread.label != thread.label
        self.end_block()
        g = ink(self.console)
        if new_speaker or self.flat:
            self._header(thread, thinking=kind == "thinking")
        self._thread = thread
        self._kind = kind
        if kind == "thinking":
            self._gutter = Gutter(self.console, self._prefix(thread, g.rail_dim), style="loom.think")
        else:
            self._gutter = Gutter(self.console, self._prefix(thread, g.rail))

    def is_open(self) -> bool:
        """True while a token block is open — i.e. :meth:`text` will actually
        draw something.

        Every other drawing path (a tool call, an approval prompt, an aside)
        ends the block, and an approval prompt is drawn from a tool worker
        thread while the stream loop is mid-message. A streaming caller must
        therefore ask the weave what is open rather than trust its own
        bookkeeping, or its tokens go nowhere.
        """
        return self._gutter is not None

    def _header(self, thread: Thread, *, thinking: bool = False) -> None:
        self._settle()
        g = ink(self.console)
        line = self._ancestors(thread)
        line.append(f"{g.think if thinking else g.bullet} ", style=f"{thread.style}.b")
        line.append(thread.label, style=f"{thread.style}.b")
        if thinking:
            line.append(" thinking", style="loom.think")
        # The badge is the expensive information — print it once per thread per
        # turn, then let the rail colour carry the identity.
        if thread.model and thread.label not in self._announced:
            line.append("  ")
            line.append_text(model_badge(self.console, thread.model, thread.is_local))
        self._announced.add(thread.label)
        self.console.print(line)

    def text(self, chunk: str) -> None:
        self._settle()
        if self._gutter:
            self._gutter.write(chunk)

    def markdown(self, text: str) -> None:
        """Non-streamed prose: render it as Markdown inside the rail."""
        from rich.markdown import Markdown

        if not self._gutter:
            return
        try:
            self._gutter.block(Markdown(text))
        except Exception:
            self._gutter.write(text)

    def end_block(self) -> None:
        self._settle()
        if self._gutter:
            self._gutter.flush()
            if not self._gutter._fresh:
                self.console.print()
        self._gutter = None
        self._kind = ""

    # -- structure
    def tool_call(
        self,
        thread: Thread,
        name: str,
        detail: str = "",
        *,
        target: Text | None = None,
    ) -> None:
        """A tool call, hanging off the caller's rail."""
        self._settle()
        self.end_block()
        g = ink(self.console)
        # A call from a thread that wasn't the one just speaking announces
        # itself: the rail alone identifies a caller you can still see the
        # header for, but a long run of calls scrolls its header away.
        if not self._thread or self._thread.label != thread.label:
            self._header(thread)
        self._thread = thread
        if self.flat:
            line = Text("  ")
        else:
            line = self._ancestors(thread)
            line.append(g.branch, style=thread.style)
            line.append(f"{g.h} ", style="loom.line")
        line.append(name, style="loom.tool")
        if detail:
            line.append(f"  {detail}", style="loom.muted")
        if target is not None:
            line.append(f"  {g.arrow} ", style="loom.line")
            line.append_text(target)
        self.console.print(line)
        self._thread = thread

    def tool_result(self, thread: Thread, text: str, *, extra_lines: int = 0, bad: bool = False) -> None:
        self._settle()
        g = ink(self.console)
        line = self._prefix(thread, g.rail) if not self.flat else Text("  ")
        line.append(f" {g.result} ", style="loom.line")
        line.append(text, style="loom.bad" if bad else "loom.muted")
        if extra_lines > 0:
            line.append(f" {g.ellipsis} +{extra_lines} lines", style="loom.line")
        self.console.print(line)

    def block(self, thread: Thread, renderable: RenderableType) -> None:
        """A full renderable (a diff, a table) indented under a thread."""
        self._settle()
        self.end_block()
        Gutter(self.console, self._prefix(thread, ink(self.console).rail)).block(renderable)

    def close_thread(self, thread: Thread, summary: Text | str = "") -> None:
        """Tie off a delegated thread — its work is done and its context is
        discarded. Only the summary crosses back to the caller, which is the
        whole point of delegating, so it gets a line of its own."""
        if thread.depth == 0:
            return
        self.end_block()
        g = ink(self.console)
        line = self._ancestors(thread)
        line.append(f"{g.corner}{g.h} ", style=thread.style)
        line.append_text(summary if isinstance(summary, Text) else Text(summary, style="loom.muted"))
        self.console.print(line)
        self._thread = None
        del self._stack[thread.depth :]

    def aside(self, message: Text | str, *, style: str = "loom.muted") -> None:
        """A line that belongs to no thread — interrupts, mode changes."""
        self._settle()
        self.end_block()
        self.console.print(message if isinstance(message, Text) else Text(message, style=style))

    def close(self, summary: Text | None = None, *, ok: bool = True) -> None:
        """Seal the weave with the turn's receipt."""
        self._settle()
        self.end_block()
        g = ink(self.console)
        line = Text()
        line.append(f"{g.corner}{g.h} ", style="loom.line")
        line.append(f"{g.ok if ok else g.stop} ", style="loom.good" if ok else "loom.warn")
        if summary is not None:
            line.append_text(summary)
        self.console.print(line)
        self._thread = None


# ---------------------------------------------------------------------------
# Composed layouts used by more than one surface
# ---------------------------------------------------------------------------


def fleet_table(console: Console, roles, *, header: bool = True) -> Table:
    """The fleet roster: role, where it runs, which model.

    ``roles`` is a sequence of ``(role, model, is_local)`` or
    ``(role, model, is_local, note)``.
    """
    out = table(

        console,
        ("role" if header else "", {"style": "loom.text", "no_wrap": True}),
        ("", {"justify": "center", "no_wrap": True}),
        ("model" if header else "", {"overflow": "ellipsis", "no_wrap": True}),
        ("" if header else "", {"style": "loom.muted", "overflow": "ellipsis"}),
    )
    out.show_header = header
    for role, model, is_local, *rest in roles:
        out.add_row(
            Text(role, style="loom.bright" if role == "orchestrator" else "loom.text"),
            where(console, is_local),
            Text(model, style="loom.local" if is_local else "loom.cloud"),
            Text(rest[0] if rest else "", style="loom.muted"),
        )
    return out


def stack(*renderables: RenderableType) -> Group:
    return Group(*[r for r in renderables if r is not None])


class Working:
    """A transient "thinking" indicator for the gap before the first token.

    Rich's Live region and ordinary printing cannot share the terminal, so
    this stops itself the moment anything else is about to draw — call
    :meth:`stop` before every print path. It is a no-op off a terminal, where
    an animation would just be thousands of escape codes in a log file.
    """

    def __init__(self, console: Console, message: str = "working") -> None:
        self.console = console
        self.message = message
        self._status = None

    def start(self) -> None:
        if self._status is not None or not self.console.is_terminal:
            return
        from rich.spinner import Spinner

        g = ink(self.console)
        spinner = Spinner("dots", text=Text(f" {self.message}…", style="loom.think"))
        spinner.frames = list(g.spinner)
        try:
            self._status = self.console.status(spinner)
            self._status.start()
        except Exception:
            self._status = None  # never let decoration break a turn

    def stop(self) -> None:
        if self._status is None:
            return
        status, self._status = self._status, None
        try:
            status.stop()
        except Exception:
            pass

    def __enter__(self) -> Working:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
