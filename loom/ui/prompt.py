"""The input line and the status bar.

Everything prompt_toolkit touches lives here: the palette translated into a
prompt_toolkit ``Style``, the caret, the completer, the key bindings, and the
status line that sits under the box.

The status line is the session's instrument cluster — what is driving, where
it runs, which mode is armed, and what it has cost so far. It is the one
surface that is always on screen, so it earns its width: a mode that can
change files without asking is inverse-video and loud; free local work is
stated as free.
"""

from __future__ import annotations

import shutil

from loom.ui.glyphs import glyphs
from loom.ui.theme import Palette, active_theme_name, resolve_palette

# prompt_toolkit style class -> palette attribute. Kept declarative so a new
# theme needs no changes here.
#
# These are deliberately `pt.*`, not `loom.*`: prompt_toolkit and Rich are two
# separate style systems, and sharing a prefix made it look as though a name
# defined for one would resolve in the other. It doesn't.
_STYLE_MAP = {
    "pt.local": "local",
    "pt.cloud": "cloud",
    "pt.muted": "muted",
    "pt.line": "line",
    "pt.text": "text",
    "pt.warp": "warp",
    "pt.good": "good",
    "pt.warn": "warn",
    "pt.bad": "bad",
    "pt.caret": "warp",
}

# Modes that can act without asking get a loud inverse badge; the calmer ones
# stay quiet. The colour is the warning, not the wording.
_MODE_STYLE = {
    "yolo": "bad",
    "accept-edits": "warn",
    "plan": "warp",
    "airgap": "good",
    "local": "local",
    "vim": "muted",
    "default": "line",
}


def pt_style(theme_name: str):
    """A prompt_toolkit Style derived from the active Loom palette."""
    from prompt_toolkit.styles import Style

    if theme_name == "mono":
        rules = {"bottom-toolbar": "noinherit", "pt.caret": "bold"}
        rules.update({name: "noinherit" for name in _STYLE_MAP})
        rules.update({f"pt.mode.{mode}": "reverse bold" for mode in _MODE_STYLE})
        rules["pt.caret"] = "bold"
        return Style.from_dict(rules)

    palette: Palette = resolve_palette(theme_name)
    rules = {"bottom-toolbar": f"noinherit fg:{palette.muted} bg:default"}
    for name, attr in _STYLE_MAP.items():
        rules[name] = f"fg:{getattr(palette, attr)} bg:default"
    # Inverse badges: the mode's colour becomes the background.
    ink = "#0F1117" if palette.dark else "#FFFFFF"
    for mode, attr in _MODE_STYLE.items():
        colour = getattr(palette, attr)
        rules[f"pt.mode.{mode}"] = f"bold fg:{ink if attr != 'line' else palette.muted} bg:{colour}"
    rules["pt.caret"] = f"bold fg:{palette.warp} bg:default"
    return Style.from_dict(rules)


def caret(theme_name: str, symbol: str = "") -> list[tuple[str, str]]:
    """The input caret. ``❯`` in the accent colour, or whatever the user set
    as ``ui.prompt_symbol``."""
    g = glyphs(theme_name != "ascii")
    return [("class:pt.caret", f"{symbol or g.prompt} ")]


def status_line(state: dict, *, theme_name: str = "loom", width: int = 0) -> list[tuple[str, str]]:
    """The bottom status bar, as prompt_toolkit fragments.

    ``state`` comes from :func:`loom.ui.repl._toolbar_state`.

    A status line that wraps is worse than one that says less, so the parts
    are assembled in priority order and the tail is dropped until it fits.
    What survives to the narrowest terminal is what you cannot afford to be
    wrong about: which model is driving, and whether a mode is armed that
    lets it act without asking.
    """
    g = glyphs(theme_name != "ascii")
    sep = ("class:pt.line", f" {g.dot} ")
    width = width or shutil.get_terminal_size((80, 24)).columns

    where = "pt.local" if state["is_local"] else "pt.cloud"
    model = [(f"class:{where}", f"{g.local if state['is_local'] else g.cloud} {state['model']}")]

    modes: list[tuple[str, str]] = []
    for mode in state.get("modes") or []:
        klass = mode if mode in _MODE_STYLE else "default"
        modes.append(("", " "))
        modes.append((f"class:pt.mode.{klass}", f" {mode} "))

    tags = state.get("local_tags") or []
    fleet: list[tuple[str, str]] = []
    if tags:
        shown = tags[0] if len(tags) == 1 else f"{tags[0]} +{len(tags) - 1}"
        fleet = [sep, ("class:pt.local", f"{g.local} {shown}")]

    cost = state.get("cost") or 0.0
    approx = "~" if state.get("estimated") else ""
    money = [("class:pt.muted" if not cost else "class:pt.text", f"{approx}${cost:.3f}")]

    share = state.get("local_share") or 0.0
    free = [("class:pt.local", f"{share:.0%} free"), sep] if share else []

    hint = [sep, ("class:pt.muted", f"shift+tab {g.dot} /help")]

    # Widest first; each fallback drops the least load-bearing group left.
    for left_parts, right_parts in (
        ([model, fleet, modes], [free, money, hint]),
        ([model, fleet, modes], [free, money]),
        ([model, modes], [money]),
        ([model, modes], []),
        ([modes], []),
    ):
        left = [("", " ")] + [f for part in left_parts for f in part]
        right = [f for part in right_parts for f in part] + [("", " ")]
        pad = width - _len(left) - _len(right)
        if pad >= 1:
            return left + [("", " " * pad)] + right
    # Narrower than the modes alone: truncate rather than wrap.
    return [("", " "), ("class:pt.muted", state["model"][: max(0, width - 2)])]


def _len(fragments) -> int:
    return sum(len(text) for _, text in fragments)


def make_prompt_session(session=None):
    """The prompt_toolkit session: history, slash completion, key bindings."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    from loom.core import settings as settings_mod
    from loom.ui import slash

    history_file = settings_mod.cfg.USER_CONFIG_DIR / "history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    completer = WordCompleter([f"/{name}" for name in slash._REGISTRY], sentence=True)
    kb = KeyBindings()
    if session is not None:

        @kb.add("s-tab")
        def _cycle(event) -> None:
            session.cycle_approval_mode()
            event.app.invalidate()  # the status line has to catch up

    theme_name = active_theme_name(session.settings.ui) if session is not None else "loom"
    return PromptSession(
        history=FileHistory(str(history_file)),
        completer=completer,
        key_bindings=kb,
        style=pt_style(theme_name),
        include_default_pygments_style=False,
    )
