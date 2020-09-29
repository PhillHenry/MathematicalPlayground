import numpy as np


def mean_of(xs):
    totals = np.sum(xs, axis=0)
    return totals / np.shape(xs)[0]


def covariance_of(xs):
    r"""
    See https://jakevdp.github.io/blog/2015/07/06/model-complexity-myth/ for the mathematical
    underpinning of conditioning a matrix.

    :param xs: array-like object of values
    :return: the covariance matrix
    """
    averaged = xs - mean_of(xs)
    m = np.dot(averaged.T, averaged)
    n = np.shape(m)[0]
    conditioned = m + (np.eye(n) * 1e-10)
    return np.dot(conditioned, np.linalg.inv(conditioned))


