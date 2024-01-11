from utils.rolling_stats import update_min


def test_update_min():
    xs = {}
    key = 5
    update_min(xs, key, 1)
    assert xs[key] == 1
    update_min(xs, key, 1, 0)
    assert xs[key] == 0
