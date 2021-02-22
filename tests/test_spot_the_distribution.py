import distributions.spot_the_distribution as to_test


def test_compare_gaussian_to_exponentials():
    n = 10
    comparison, xs, ys = to_test.compare_gaussian_to_exponentials(n)
    assert(len(xs) == n)
    assert(len(ys) == n)
