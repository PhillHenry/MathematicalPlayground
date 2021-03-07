import numpy as np
from distributions.spot_the_distribution import *


def test_compare_gaussian_to_exponentials():
    n = 10
    over = np.arange(1, 20, 0.2)
    comparison, xs, ys = make_comparison(n, run(using_t_test, over))
    print("comparison = {}, xs = {}, ys = {}".format(comparison, xs, ys))
    assert(len(xs) == n)
    assert(len(ys) == n)


def test_compare_kolmogorov_smirnov():
    np.random.seed(12345678)
    n = 10000
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    z = np.random.normal(1.1, 0.9, n)
    assert(kolmogorov_smirnov_comparison(x, y) > 0.05)


