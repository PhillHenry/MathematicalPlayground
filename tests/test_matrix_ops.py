from matrix import ops
import numpy as np
import pytest


def test_invert_non_square_matrix():
    with pytest.raises(Exception) as e:
        ops.condition_and_invert(np.empty([5, 7]))


def test_invert_square_matrix():
    n = 11
    inverted = ops.condition_and_invert(np.empty([n, n]))
    assert np.shape(inverted) == (n, n)

