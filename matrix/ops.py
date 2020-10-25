import numpy as np


def condition_and_invert(m):
    shape = np.shape(m)
    width = shape[0]
    height = shape[1]
    if width != height:
        raise Exception("{} x {} matrix is not square. Cannot invert.".format(width, height))
    return np.linalg.inv(m + (np.eye(width) * 1e-20))
