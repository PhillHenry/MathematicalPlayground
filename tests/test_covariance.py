from math import pi
import numpy as np
import distributions.covariance as to_test


def test_mean_of_row_vector():
    xs = np.asmatrix([1, 2, 3, 4])
    assert(np.array_equal(to_test.mean_of(xs), xs))


def test_covariance_of_row_vector():
    xs = np.asmatrix([1, 2, 3, 4])
    m = to_test.covariance_of(xs)
    assert(np.shape(m) == (len(xs), len(xs)))
    for i in range(len(xs)):
        assert(m[i, i] == 1.)


