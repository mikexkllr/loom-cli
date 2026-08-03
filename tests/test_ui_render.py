"""The UI's structural guarantees.

These test the properties that break silently and everywhere at once — a style
name that doesn't resolve, a rail that loses its gutter, a glyph that only
exists in UTF-8 — rather than pinning down exact spacing, which is meant to
keep changing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console

from loom.core.settings import UISettings
from loom.ui import banner, render
from loom.ui import prompt as prompt_mod
from loom.ui.glyphs import ASCII, UNICODE
from loom.ui.render import Weave
from loom.ui.theme import THEME_NAMES, active_theme_name, make_console, styles_for, theme_of

LOOM_ROOT = Path(__file__).resolve().parent.parent / "loom"
# Style names as they appear in markup ("[loom.warn]") and as string literals
# ("style=\"loom.warn\"", f"loom.t{i}" excluded — those are generated).
_STYLE_RE = re.compile(r'["\[/](loom\.[a-z][a-z.]*)["\]]')


def _console(theme: str = "loom", width: int = 80) -> Console:
    console = make_console(UISettings(theme=theme))
    console.width = width
    return console


def _capture(console: Console, fn) -> str:
    with console.capture() as cap:
        fn()
    return cap.get()


# --------------------------------------------------------------------- themes


def _style_names_used() -> set[str]:
    names: set[str] = set()
    for path in LOOM_ROOT.rglob("*.py"):
        for match in _STYLE_RE.finditer(path.read_text(encoding="utf-8")):
            names.add(match.group(1))
    return names


def test_every_style_name_in_the_source_resolves_in_every_theme():
    """A style Rich can't parse raises MissingStyle at print time, so a name
    that only one theme defines is a crash waiting for a `/theme` change."""
    used = _style_names_used()
    assert "loom.warp" in used, "the scan should be finding style names at all"
    for theme in THEME_NAMES:
        console = _console(theme)
        for name in sorted(used):
            console.get_style(name)  # raises MissingStyle if undefined


def test_every_theme_defines_the_same_style_names():
    reference = set(styles_for("loom"))
    for theme in THEME_NAMES:
        assert set(styles_for(theme)) == reference, f"{theme} has a different style set"


def test_a_plain_console_is_adopted_rather_than_crashing():
    """Hooks, tests and embedders build their own Console. Loom's styles have
    to arrive with the first primitive that touches it."""
    plain = Console()
    with pytest.raises(Exception):
        plain.get_style("loom.warp")
    theme_of(plain)
    plain.get_style("loom.warp")  # now resolves


def test_no_color_forces_mono(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert active_theme_name(UISettings(theme="loom")) == "mono"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert active_theme_name(UISettings(theme="phosphor")) == "mono"


def test_legacy_theme_names_still_load():
    for old in ("auto", "dark", "light"):
        assert UISettings(theme=old).theme == old
        make_console(UISettings(theme=old))


def test_unknown_theme_is_rejected_at_the_settings_boundary():
    with pytest.raises(ValueError):
        UISettings(theme="solarized-teal")


# --------------------------------------------------------------------- glyphs


def test_ascii_glyph_set_is_pure_ascii():
    """The fallback exists for terminals that mangle UTF-8; a stray box-drawing
    character in it defeats the whole point."""
    for field in ASCII.__dataclass_fields__:
        value = getattr(ASCII, field)
        for text in value if isinstance(value, tuple) else [value]:
            assert text.isascii(), f"ASCII.{field} = {text!r} is not ASCII"


def test_glyph_sets_cover_the_same_names():
    assert set(UNICODE.__dataclass_fields__) == set(ASCII.__dataclass_fields__)


def test_ascii_mode_renders_without_box_drawing(monkeypatch):
    monkeypatch.setenv("LOOM_ASCII", "1")
    console = _console()
    weave = Weave(console)
    thread = weave.thread("orchestrator", "claude-sonnet-5", is_local=False)

    def draw():
        weave.open(thread)
        weave.text("hello there\n")
        weave.tool_call(thread, "read_file", "a.py")
        weave.tool_result(thread, "12 lines")
        weave.close()

    out = _capture(console, draw)
    assert out.isascii(), f"non-ASCII leaked through: {out!r}"


# ---------------------------------------------------------------------- weave


def test_every_line_of_a_block_carries_its_rail():
    console = _console(width=40)
    weave = Weave(console)
    thread = weave.thread("orchestrator", "claude-sonnet-5", is_local=False)
    out = _capture(
        console,
        lambda: (weave.open(thread), weave.text("word " * 40), weave.end_block()),
    )
    body = [line for line in out.splitlines() if line and "orchestrator" not in line]
    assert len(body) > 1, "the text should have wrapped onto several lines"
    assert all(line.startswith("│") for line in body), body


def test_wrapping_never_splits_a_word():
    console = _console(width=30)
    weave = Weave(console)
    thread = weave.thread("orchestrator")
    # Fed one character at a time, the way tokens actually arrive.
    words = ["antidisestablishmentarianism", "supercalifragilistic", "pneumonoultramicroscopic"]
    text = " ".join(words) + " "

    def draw():
        weave.open(thread)
        for char in text:
            weave.text(char)
        weave.end_block()

    out = _capture(console, draw)
    stripped = " ".join(line.lstrip("│ ") for line in out.splitlines())
    for word in words:
        assert word in stripped, f"{word} was split across a wrap"


def test_a_subagent_rail_is_indented_under_its_caller():
    console = _console()
    weave = Weave(console)
    orch = weave.thread("orchestrator", "claude-sonnet-5", is_local=False, depth=0)
    out = _capture(
        console,
        lambda: (
            weave.open(orch),
            weave.text("delegating\n"),
            weave.open(weave.thread("reviewer", "qwen3.5:9b", is_local=True, depth=1)),
            weave.text("found two issues\n"),
            weave.end_block(),
        ),
    )
    lines = out.splitlines()
    assert any(line.startswith("│ delegating") for line in lines)
    assert any(line.startswith("╎ ") and "found two issues" in line for line in lines)


def test_a_thread_announces_itself_once_per_turn():
    console = _console()
    weave = Weave(console)

    def draw():
        for _ in range(3):
            weave.open(weave.thread("orchestrator", "claude-sonnet-5", is_local=False))
            weave.text("chunk\n")
            weave.open(weave.thread("reviewer", "qwen3.5:9b", is_local=True, depth=1))
            weave.text("chunk\n")
        weave.end_block()

    out = _capture(console, draw)
    # The name repeats when the speaker changes; the model badge does not.
    assert out.count("claude-sonnet-5") == 1
    assert out.count("qwen3.5:9b") == 1
    assert out.count("orchestrator") == 3


def test_reset_clears_thread_colours_and_headers():
    console = _console()
    weave = Weave(console)
    weave.thread("reviewer", depth=1)
    assert weave._slots
    weave.reset()
    assert not weave._slots and not weave._stack and not weave._announced


def test_flat_mode_drops_the_rails():
    """`ui.weave: false` is for piping a session into a file, where a gutter
    of box-drawing characters is just noise."""
    console = _console()
    weave = Weave(console, flat=True)
    thread = weave.thread("orchestrator", "claude-sonnet-5", is_local=False)
    out = _capture(
        console,
        lambda: (weave.open(thread), weave.text("hello\n"), weave.end_block()),
    )
    assert "│" not in out and "╎" not in out


def test_local_and_cloud_badges_are_distinct_and_coloured():
    console = _console()
    local = render.where(console, True)
    cloud = render.where(console, False)
    assert local.plain != cloud.plain
    assert local.style == "loom.local" and cloud.style == "loom.cloud"


# --------------------------------------------------------------- status line


def test_status_line_fits_the_terminal_width():
    state = {
        "model": "claude-sonnet-5",
        "is_local": False,
        "local_tags": ["qwen3.5:9b", "qwen3.6:27b"],
        "modes": ["plan", "local"],
        "cost": 1.234,
        "estimated": False,
        "local_share": 0.62,
        "turns": 4,
    }
    for width in (40, 80, 120, 200):
        line = prompt_mod.status_line(state, width=width)
        assert sum(len(text) for _, text in line) <= width, f"overflowed at {width}"


def test_status_line_reports_model_mode_and_cost():
    state = {
        "model": "claude-sonnet-5",
        "is_local": False,
        "local_tags": ["qwen3.5:9b"],
        "modes": ["yolo"],
        "cost": 0.5,
        "estimated": True,
        "local_share": 0.4,
        "turns": 1,
    }
    text = "".join(t for _, t in prompt_mod.status_line(state, width=120))
    assert "claude-sonnet-5" in text
    assert "yolo" in text
    assert "~$0.500" in text  # the ~ has to survive: it means "estimated"
    assert "40% free" in text


def test_every_theme_builds_a_prompt_toolkit_style():
    for theme in THEME_NAMES:
        prompt_mod.pt_style(theme)


# -------------------------------------------------------------------- banner


def test_wordmark_falls_back_to_ascii(monkeypatch):
    monkeypatch.setenv("LOOM_ASCII", "1")
    assert banner.wordmark(_console()).plain.isascii()


def test_wordmark_lines_are_equal_width():
    """A ragged wordmark reads as a rendering bug, not a logo."""
    for art in (banner.WORDMARK, banner.ASCII_WORDMARK):
        assert len({len(line) for line in art}) == 1, art


# ------------------------------------------------------------------- working


def test_working_indicator_is_silent_off_a_terminal():
    """A Live animation in a pipe or a CI log is thousands of escape codes and
    zero information."""
    console = _console()
    assert not console.is_terminal
    working = render.Working(console)
    out = _capture(console, lambda: (working.start(), working.stop()))
    assert out == ""


def test_the_weave_retires_the_indicator_before_it_draws():
    """Rich's Live region and an ordinary print cannot share the terminal, so
    every drawing path has to stop the spinner first."""
    console = _console()
    weave = Weave(console)
    weave.working = render.Working(console)
    stopped: list[bool] = []
    weave.working.stop = lambda: stopped.append(True)  # type: ignore[method-assign]

    thread = weave.thread("orchestrator", "claude-sonnet-5", is_local=False)
    for draw in (
        lambda: weave.open(thread),
        lambda: weave.text("hi"),
        lambda: weave.tool_call(thread, "read_file", "a.py"),
        lambda: weave.tool_result(thread, "ok"),
        lambda: weave.aside("note"),
        lambda: weave.close(),
    ):
        stopped.clear()
        _capture(console, draw)
        assert stopped, f"{draw} drew without retiring the indicator"


def test_a_broken_indicator_never_breaks_a_turn(monkeypatch):
    console = _console()
    monkeypatch.setattr(type(console), "is_terminal", property(lambda self: True))
    monkeypatch.setattr(console, "status", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    working = render.Working(console)
    working.start()  # must swallow
    working.stop()


# ------------------------------------------------------------------ contrast


def _relative_luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# Loom does not paint the background — it inherits the terminal's. So every
# colour has to survive the range of backgrounds people actually run, not one
# assumed value. These are the common dark/light terminal defaults.
_BACKGROUNDS = {
    "loom": ("#000000", "#1e1e1e", "#282c34", "#1a1b26", "#2d2d2d"),
    "loom-light": ("#ffffff", "#fdf6e3", "#f5f5f5"),
    "phosphor": ("#000000", "#0a1408", "#111111"),
}

# Borders and rules are meant to be quiet, but a frame nobody can see reads as
# a rendering bug — the panel title ends up floating in space.
_MINIMUM = {"line": 1.9, "think": 2.4, "muted": 3.3, "text": 6.0, "bright": 8.0}


@pytest.mark.parametrize("theme", sorted(_BACKGROUNDS))
def test_palette_stays_legible_on_real_terminal_backgrounds(theme):
    from loom.ui.theme import resolve_palette

    palette = resolve_palette(theme)
    for attr, minimum in _MINIMUM.items():
        colour = getattr(palette, attr)
        for background in _BACKGROUNDS[theme]:
            got = _contrast(colour, background)
            assert got >= minimum, (
                f"{theme}.{attr} ({colour}) is {got:.2f}:1 on {background}, "
                f"needs {minimum}:1 — it will look invisible or broken there"
            )


@pytest.mark.parametrize("theme", sorted(_BACKGROUNDS))
def test_the_signal_colours_stay_loud(theme):
    """warm=local / cool=cloud is the one thing the eye must catch, so those
    two — and the accent — get a stricter floor than the furniture."""
    from loom.ui.theme import resolve_palette

    palette = resolve_palette(theme)
    for attr in ("local", "cloud", "warp", "good", "warn", "bad"):
        for background in _BACKGROUNDS[theme]:
            got = _contrast(getattr(palette, attr), background)
            assert got >= 3.5, f"{theme}.{attr} is only {got:.2f}:1 on {background}"


@pytest.mark.parametrize("theme", sorted(_BACKGROUNDS))
def test_thread_rail_colours_are_distinguishable(theme):
    """Two subagents streaming at once must not share a rail colour."""
    from loom.ui.theme import resolve_palette

    threads = resolve_palette(theme).threads
    assert len(set(threads)) == len(threads), "duplicate thread colour"
    assert len(threads) >= 5, "too few rails before the cycle repeats"
