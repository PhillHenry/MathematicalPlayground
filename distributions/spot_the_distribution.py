import numpy as np
from scipy import stats


def compare_tstats(a, b, N):
    t2, p2 = stats.ttest_ind(a,b)
    print("t = " + str(t2))
    print("p = " + str(p2)) # ie, probability of rejecting the null hypothesis that the distributions are different


if __name__ == "__main__":
    n = 10
    gaussians = np.random.normal(10, 1, n)
    print(gaussians)
    binomials = np.random.binomial(10, 0.5, n)
    print(binomials)

    exponentials = np.random.exponential(10, n)
    print(exponentials)

    compare_tstats(gaussians, exponentials, n)
