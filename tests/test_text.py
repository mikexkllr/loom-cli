"""Reading text out of a model message, whichever shape its content took."""

from types import SimpleNamespace

from loom.core.text import content_text, message_text


def test_a_plain_string_is_itself():
    assert content_text("hello") == "hello"
    assert content_text("") == "" and content_text(None) == ""


def test_a_block_list_yields_its_text_blocks_only():
    # What Anthropic returns whenever a reply also calls a tool or thinks.
    blocks = [
        {"type": "thinking", "thinking": "let me see"},
        {"type": "text", "text": "I'll map the repo. "},
        {"type": "tool_use", "id": "t1", "name": "task", "input": {"subagent_type": "explorer"}},
        {"type": "text", "text": "Done."},
        "tail",
    ]
    assert content_text(blocks) == "I'll map the repo. Done.tail"
    assert content_text([{"type": "tool_use", "id": "t1", "name": "task", "input": {}}]) == ""


def test_message_text_reads_objects_and_role_pairs():
    assert message_text(SimpleNamespace(content=[{"type": "text", "text": "hi"}])) == "hi"
    assert message_text(("user", "do the thing")) == "do the thing"
    assert message_text("bare") == "bare"
