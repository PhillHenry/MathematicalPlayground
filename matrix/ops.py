import numpy as np


def condition_and_invert(m):
    r"""
    See https://jakevdp.github.io/blog/2015/07/06/model-complexity-myth/ for the mathematical
    underpinning of conditioning a matrix.
    """
    shape = np.shape(m)
    width = shape[0]
    height = shape[1]
    if width != height:
        raise Exception("{} x {} matrix is not square. Cannot invert.".format(width, height))
    return np.linalg.inv(m + (np.eye(width) * 1e-20))
