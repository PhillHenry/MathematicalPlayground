import numpy as np


def row_mean_of(xs):
    totals = np.sum(xs, axis=1)
    return totals / np.shape(xs)[1]


def covariance_of(xs):
    r"""
    :param xs: array-like object of values
    :return: the covariance matrix
    """
    averaged = xs - row_mean_of(xs)
    return squared_with_bessel(averaged)


def squared_with_bessel(averaged):
    m = np.dot(averaged, averaged.T)
    n = np.shape(averaged)[1]
    return m / (n - 1)


