import numpy as np
from scipy import stats


def compare_tstats(a, b):
    t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    return t2, p2


def compare_gaussian_to_exponentials(n=1000):
    gaussians = np.random.normal(10, 1, n)
    # print(gaussians)

    exponentials = np.random.exponential(10, n)
    # print(exponentials)

    return compare_tstats(gaussians, exponentials), gaussians, exponentials


if __name__ == "__main__":
    n_trials = 100
    trials = list(map(lambda _: compare_gaussian_to_exponentials(), range(0, n_trials)))
    m = np.asarray([*trials], object)
    metrics, xs, ys = np.transpose(m)
    tps = np.array([*metrics])
    # print(tps)
    mean_p = np.mean(tps[:, 1])
    print("mean p = {}".format(mean_p))  # higher means more likely to be the same distribution

