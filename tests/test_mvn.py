from distributions import multvariate_normal
from math import pi


def test_gaussian():
    assert multvariate_normal.gaussian([0], 0, 1) == 1. / (2 * pi) ** 0.5




