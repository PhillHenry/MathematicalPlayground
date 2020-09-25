import numpy as np


def mean_of(xs):
    totals = np.sum(xs, axis=0)
    return totals / np.shape(xs)[0]


def covariance_of(xs):
    r"""
    :param xs: array-like object of values
    :return: the covariance matrix
    """
    averaged = xs - mean_of(xs)
    return np.dot(averaged, averaged.T)


