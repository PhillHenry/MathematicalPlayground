from models.lin_reg_1hot_vs_dummy import deltas
import numpy as np


def test_deltas():
    ds = deltas([0, 1, 1, 2, 3, 5, 8, 13])
    assert np.allclose(ds, [1, 0, 1, 1, 2, 3, 5])