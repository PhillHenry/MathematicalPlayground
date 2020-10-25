import numpy as np


def lower_left(n):
    r"""
    :return: n x n matrix with the lower left values of 1.0
    """
    a = np.empty([n, n])
    a.fill(0.)
    for i in range(n):
        for j in range(i + 1):
            a[i, j] = 1.
    return a
