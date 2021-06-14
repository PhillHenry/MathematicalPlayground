from models.lin_reg_1hot_vs_dummy import deltas, skip_every, num_non_increasing
import numpy as np


def test_deltas():
    xs = [0, 1, 1, 2, 3, 5, 8, 13]
    ds = deltas(xs)
    assert np.allclose(ds, [1, 0, 1, 1, 2, 3, 5])
    assert len(ds) == len(xs) - 1


def test_skip_every():
    xs = skip_every(4, list(range(12)))
    assert np.allclose(xs, [0, 1, 2, 4, 5, 6, 8, 9, 10])


def test_num_non_increasing():
    xs = [-2.507, -1.507, -0.507, 0.493, 1.493, -3.821, -2.821, -1.821, -0.821, 0.179, -1.422, -0.422, 0.578, 1.578, 2.578, -1.449, -0.449, 0.551, 1.551, 2.551]
    assert num_non_increasing(xs, 5) == 0
