import numpy as np


def row_mean_of(xs):
    totals = np.sum(xs, axis=1)
    return totals / np.shape(xs)[1]


def covariance_of(xs):
    r"""
    See https://jakevdp.github.io/blog/2015/07/06/model-complexity-myth/ for the mathematical
    underpinning of conditioning a matrix.

    :param xs: array-like object of values
    :return: the covariance matrix
    """
    averaged = xs - row_mean_of(xs)
    m = np.dot(averaged, averaged.T)
    n = np.shape(xs)[1]
    return m / (n - 1)  # np.dot(conditioned, np.linalg.inv(conditioned))


