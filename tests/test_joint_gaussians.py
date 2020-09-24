from distributions import joint_gaussians
from math import pi
import numpy as np


def test_gaussian():
    assert joint_gaussians.gaussian([0], 0, 1) == 1. / (2 * pi) ** 0.5


def test_joint():
    n = 10
    support = np.arange(0, 10)
    xs, ys = np.meshgrid(support, support)
    assert np.shape(xs) == (n, n)
    assert np.shape(ys) == (n, n)

