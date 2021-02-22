import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compare_tstats(a, b):
    t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    return t2, p2


def compare_gaussian_to_exponentials(n=1000):
    gaussians = np.random.normal(10, 1, n)

    exponentials = np.random.exponential(10, n)

    return compare_tstats(gaussians, exponentials), gaussians, exponentials


def plot(ps, xs, ys):
    mean_p = np.mean(ps)
    print("mean p = {}".format(mean_p))  # higher means more likely to be the same distribution
    xs = np.hstack(xs)
    ys = np.hstack(ys)
    plt.subplot(211)
    plt.title("Two distributions with probability {:.2f} of being the same using t-tests".format(mean_p))
    plt.hist(xs, density=True, bins=30)
    plt.subplot(212)
    plt.hist(ys, density=True, bins=30)
    plt.show()


if __name__ == "__main__":
    n_trials = 100
    trials = list(map(lambda _: compare_gaussian_to_exponentials(), range(0, n_trials)))
    m = np.asarray([*trials], object)
    metrics, xs, ys = np.transpose(m)
    tps = np.array([*metrics])
    plot(tps[:, 1], xs, ys)
