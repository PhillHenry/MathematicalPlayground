import numpy as np
import math as math
import distributions.covariance as to_test

xs = [1, 2, 3, 4]
v = np.asmatrix(xs)
m = np.stack((v + 1, v - 1))


def test_mean_of_col_vector():
    mean = to_test.row_mean_of(v.T)
    assert np.array_equal(mean, v.T), "actual = {}".format(mean)


def test_mean_of_row_vector():
    assert to_test.row_mean_of(v) == 2.5


def test_mean_of_matrix():
    mean = to_test.row_mean_of(m)
    assert np.array_equal(mean.T, np.asmatrix([3.5, 1.5]))


def test_covariance_of_row_vector():
    c = to_test.covariance_of(v)
    assert(np.shape(c) == (1, 1))
    for i in range(len(v)):
        variance = np.std(v, ddof=1) ** 2
        assert math.isclose(c[i, i], variance, abs_tol=1e-8)  # Think Stats, p108


def test_covariance_of_matrix():
    c = to_test.covariance_of(m)
    l = m.shape[0]
    assert(np.shape(c) == (l, l))
    assert np.allclose(c, np.cov(m)), "c\n{}\nnumpy.cov:\n{}".format(c, np.cov(m))
    for i in range(l):
        for j in range(l):
            if i != j:
                assert(c[i, j] <= c[i, i])
