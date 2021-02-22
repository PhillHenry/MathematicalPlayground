import distributions.spot_the_distribution as to_test
import numpy as np


def test_compare_gaussian_to_exponentials():
    n = 10
    comparison, xs, ys = to_test.compare_gaussian_to_exponentials(n)
    assert(len(xs) == n)
    assert(len(ys) == n)


def test_compare_kolmogorov_smirnov():
    np.random.seed(12345678)
    n = 10000
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    z = np.random.normal(1.1, 0.9, n)
    assert(to_test.compare_kolmogorov_smirnov(x, y) > 0.05)


