import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def compare_tstats(a, b):
    t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    return t2, p2


def compare_kolmogorov_smirnov(a, b):
    return ks_2samp(a, b).pvalue


def compare_gaussian_to_exponentials(fn, n=1000):
    gaussians = np.random.normal(10, 1, n)
    exponentials = np.random.exponential(10, n)
    return fn(gaussians, exponentials), gaussians, exponentials


def plot(ps, xs, ys, test):
    mean_p = np.mean(ps)
    print("mean p = {}".format(mean_p))  # higher means more likely to be the same distribution
    xs = np.hstack(xs)
    ys = np.hstack(ys)
    plt.subplot(211)
    plt.title("Two distributions with probability {:.2f} of being the same using {}".format(mean_p, test))
    plt.hist(xs, density=True, bins=30)
    plt.subplot(212)
    plt.hist(ys, density=True, bins=30)
    plt.show()


def make_comparison(n_trials, fn):
    trials = list(map(lambda _: fn(), range(0, n_trials)))
    m = np.asarray([*trials], object)
    metrics, xs, ys = np.transpose(m)
    return metrics, xs, ys


def t_test(n_trials=100):
    fn = lambda: compare_gaussian_to_exponentials(compare_tstats)
    metrics, xs, ys = make_comparison(n_trials, fn)
    tps = np.array([*metrics])
    return tps[:, 1], xs, ys


def kolmogorov_smirnov(n_trials=100):
    fn = lambda: compare_gaussian_to_exponentials(compare_kolmogorov_smirnov)
    ps, xs, ys = make_comparison(n_trials, fn)
    return ps, xs, ys


if __name__ == "__main__":
    n_trials = 100
    ps, xs, ys = t_test(n_trials)
    plot(ps, xs, ys, "t-test")
