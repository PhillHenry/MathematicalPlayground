from models.lin_reg_1hot_vs_dummy import deltas, skip_every
import numpy as np


def test_deltas():
    ds = deltas([0, 1, 1, 2, 3, 5, 8, 13])
    assert np.allclose(ds, [1, 0, 1, 1, 2, 3, 5])


def test_skip_every():
    xs = skip_every(4, list(range(12)))
    assert np.allclose(xs, [0, 1, 2, 4, 5, 6, 8, 9, 10])
