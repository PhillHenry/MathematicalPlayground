from sorting.zorder import interleave


def test_interleave():
    assert interleave(255, 0) == 43690
