from text_tools import slugify


def test_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_runs_collapse():
    assert slugify("a  --  b") == "a-b"


def test_strip_edges():
    assert slugify("--Already-Slugged--") == "already-slugged"
