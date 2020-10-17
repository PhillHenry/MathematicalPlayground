import numpy as np

import distributions.covariance as to_test

xs = [1, 2, 3, 4]
v = np.asmatrix(xs)
m = np.stack((v + 1, v - 1))


def test_mean_of_col_vector():
    assert(to_test.mean_of(v.T) == 2.5)


def test_mean_of_row_vector():
    assert(np.array_equal(to_test.mean_of(v), v))


def test_mean_of_matrix():
    assert(np.array_equal(to_test.mean_of(m), v))


def _test_covariance_of_row_vector():
    c = to_test.covariance_of(v)
    assert(np.shape(c) == (len(xs), len(xs)))
    for i in range(len(v)):
        assert(c[i, i] == 1.)


def test_covariance_of_matrix():
    c = to_test.covariance_of(m)
    l = m.shape[0]
    assert(np.shape(c) == (l, l))
    for i in range(l):
        for j in range(l):
            if i != j:
                assert(c[i, j] <= c[i, i])
