from utils import last_n


def test_last_two():
    assert last_n([1, 2, 3], 2) == [2, 3]


def test_last_all():
    assert last_n([1, 2], 2) == [1, 2]


def test_last_one():
    assert last_n([5, 6, 7], 1) == [7]
