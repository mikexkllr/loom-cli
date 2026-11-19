"""Reading text out of a model message.

A message's ``content`` is a plain string from most providers and a list of
typed blocks from Anthropic whenever a reply carries tool calls or thinking.
``str()`` of that list is its Python repr — which is what ended up on screen,
in ``/export`` files and in compaction summaries wherever a caller assumed the
string case.
"""

from __future__ import annotations

from typing import Any


def content_text(content: Any) -> str:
    """The visible text of ``content``: the string itself, or the ``text``
    blocks of a block list joined in order. Tool-use, thinking and image
    blocks contribute nothing."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content)


def message_text(message: Any) -> str:
    """:func:`content_text` of a message object, or of a ``(role, text)`` pair."""
    if isinstance(message, tuple):
        return content_text(message[1] if len(message) > 1 else "")
    return content_text(getattr(message, "content", message))
