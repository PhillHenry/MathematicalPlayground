from distributions import joint_gaussians
from math import pi


def test_gaussian():
    assert joint_gaussians.gaussian([0], 0, 1) == 1. / (2 * pi) ** 0.5




