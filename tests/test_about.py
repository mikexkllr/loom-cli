"""`/about`: the weave animation and the card it lands on.

The animation is tested through :func:`about.frame`, which is pure — the same
arguments always draw the same screen — so the cloth, the shuttle and the
reveal of the wordmark are all checkable without a terminal. What needs a
terminal is only the *playing* of it, and the one rule there is that it must
never play when nobody is watching.
"""

import pytest

pytest.importorskip("rich")
pytest.importorskip("pydantic")

from rich.console import Console

from loom.core.settings import UISettings
from loom.ui import about, banner
from loom.ui.theme import make_console


def _console(theme: str = "loom", width: int = 80) -> Console:
    console = make_console(UISettings(theme=theme))
    console.width = width
    return console


def _plain(console: Console, renderable) -> str:
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def _rows(console: Console, **kwargs) -> list[str]:
    kwargs.setdefault("version", "9.9.9")
    return about.frame(console, **kwargs).plain.split("\n")


# ------------------------------------------------------------------ the cloth


def test_cloth_grows_downwards_and_warp_hangs_below():
    """Above the shuttle is finished cloth; below it the warp is still bare."""
    rows = _rows(_console(), width=40, height=12, woven=5, shuttle=20, forward=True)
    assert len(rows) == 12
    assert all(len(row) == 40 for row in rows)
    assert set(rows[0]) <= set("▚▞")  # cloth
    assert set(rows[11]) <= set("│╎")  # bare warp
    assert rows[5].startswith("─")  # the weft being laid, left of the shuttle


def test_the_shuttle_lays_weft_behind_it_in_both_directions():
    right = _rows(_console(), width=40, height=12, woven=5, shuttle=20, forward=True)[5]
    assert right[19] == "─" and right[20] == "◆" and right[21] in "│╎"
    left = _rows(_console(), width=40, height=12, woven=5, shuttle=20, forward=False)[5]
    assert left[19] in "│╎" and left[20] == "◆" and left[21] == "─"


def test_shuttle_reverses_at_the_selvedge_and_stays_on_screen():
    width = 40
    seen = [about.shuttle_at(f, width) for f in range(about.FRAMES_PER_PASS * 3)]
    assert all(0 <= x < width for x in seen)
    first, second = seen[: about.FRAMES_PER_PASS], seen[about.FRAMES_PER_PASS : about.FRAMES_PER_PASS * 2]
    assert first[0] == 0 and first[-1] == width - 1  # left to right
    assert second[0] == width - 1 and second[-1] == 0  # and back


# --------------------------------------------------------------- the wordmark


def test_the_mark_is_revealed_only_by_cloth_that_is_finished():
    """It is woven in, not printed on top: no cloth over it, no mark."""
    early = "\n".join(_rows(_console(), width=60, height=20, woven=1, shuttle=0, forward=True))
    late = "\n".join(_rows(_console(), width=60, height=20, woven=20, shuttle=0, forward=True))
    art_row = banner.WORDMARK[1]
    assert art_row not in early
    assert art_row in late
    assert "v9.9.9" in late and about.TAGLINE in late


def test_the_mark_sits_on_clear_cloth_so_it_can_be_read():
    """The twill is drawn with the same two glyphs as the logo, so without a
    clear patch the letters vanish into their own texture."""
    rows = _rows(_console(), width=60, height=20, woven=20, shuttle=0, forward=True)
    marked = next(row for row in rows if banner.WORDMARK[1] in row)
    left, _, right = marked.partition(banner.WORDMARK[1])
    assert left.endswith("  ") and right.startswith("  ")


def test_a_narrow_screen_drops_the_caption_rather_than_wrapping_it():
    rows = _rows(_console(), width=30, height=16, woven=16, shuttle=0, forward=True)
    joined = "\n".join(rows)
    assert about.TAGLINE not in joined
    assert all(len(row) == 30 for row in rows)


# ------------------------------------------------------------------ fallbacks


def test_ascii_terminals_get_ascii_thread_glyphs(monkeypatch):
    monkeypatch.setenv("LOOM_ASCII", "1")
    rows = _rows(_console(), width=40, height=12, woven=6, shuttle=10, forward=True)
    assert "\n".join(rows).isascii()
    assert set(rows[0]) <= set("\\/")


@pytest.mark.parametrize("theme", ["loom", "loom-light", "phosphor", "mono"])
def test_every_theme_renders_a_frame_and_the_card(theme):
    """A hex gradient is computed per band, so a theme with no palette at all
    (mono) has to be handled by name — a MissingStyle here would crash
    /about."""
    console = _console(theme)
    out = _plain(console, about.frame(console, width=50, height=14, woven=7, shuttle=5, forward=True, version="1.0.0"))
    assert out.strip()
    assert _plain(console, about.card(console)).strip()


def test_frames_run_from_bare_warp_to_finished_cloth():
    console = _console()
    stills = list(about.frames(console, width=40, height=12, version="1.0.0", count=20))
    assert len(stills) == 20
    # Frame one is bare warp with the shuttle just entering the top row.
    assert set(stills[0].plain.replace("\n", "")) <= set("│╎◆")
    assert set(stills[-1].plain.split("\n")[0]) <= set("▚▞")


# ------------------------------------------------------------------- playing


def test_never_animates_when_nobody_is_watching():
    """A pipe, a CI log or a captured buffer gets thousands of escape codes
    and no benefit."""
    console = _console()
    assert console.is_terminal is False
    assert about.can_animate(console) is False
    assert about.animate(console, version="1.0.0") is False


def test_the_opt_out_and_the_tiny_terminal_both_skip_it(monkeypatch):
    from rich.console import ConsoleDimensions

    console = _console()
    monkeypatch.setattr(type(console), "is_terminal", property(lambda self: True))
    size = ConsoleDimensions(100, 40)
    monkeypatch.setattr(type(console), "size", property(lambda self: size))
    assert about.can_animate(console) is True

    monkeypatch.setenv("LOOM_NO_ANIM", "1")
    assert about.can_animate(console) is False, "LOOM_NO_ANIM is the opt-out"
    monkeypatch.delenv("LOOM_NO_ANIM")

    size = ConsoleDimensions(20, 6)  # a split pane has no room to weave
    assert about.can_animate(console) is False


def test_show_still_prints_the_card_without_animating(monkeypatch):
    console = _console()
    played = []
    monkeypatch.setattr(about, "animate", lambda *a, **k: played.append(True))
    with console.capture() as cap:
        about.show(console, cwd="/tmp/project", still=True)
    out = cap.get()
    assert played == []
    assert "loom uninstall" in out and "/tmp/project" in out


# ---------------------------------------------------------------------- card


def test_the_card_says_which_build_you_are_on():
    from loom import __version__
    from loom.core import update as update_mod

    out = _plain(_console(width=100), about.card(_console(width=100)))
    assert __version__ in out
    assert update_mod.REPO in out
    assert "source install" in out  # pytest is never a frozen build


def test_the_card_reports_a_frozen_build_as_a_binary(monkeypatch):
    from loom.core import update as update_mod

    monkeypatch.setattr(update_mod, "is_frozen", lambda: True)
    out = _plain(_console(width=100), about.card(_console(width=100)))
    assert "standalone binary" in out


def test_slash_about_is_registered_and_renders(tmp_path, capsys):
    pytest.importorskip("langchain_core")
    from loom.core import settings as st
    from loom.ui import slash
    from loom.ui.repl import Session

    session = Session(st.load_settings(tmp_path), cwd=str(tmp_path))
    session.console.width = 100
    assert slash.dispatch(session, "/about still") is True
    out = capsys.readouterr().out
    from loom import __version__

    assert __version__ in out and "about" in out
