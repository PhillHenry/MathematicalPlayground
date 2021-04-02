import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import norm
from scipy.stats import expon


def using_kolmogorov_smirnov(support):
    return compare_gaussian_to_exponentials_using(kolmogorov_smirnov_comparison, support)


def using_t_test(support):
    return compare_gaussian_to_exponentials_using(t_test_comparison, support)


def t_test_comparison(a, b):
    t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    return t2, p2


def kolmogorov_smirnov_comparison(a, b):
    return ks_2samp(a, b).pvalue


def compare_gaussian_to_exponentials_using(fn, support=np.arange(1, 20, 0.2)):
    gaussians = norm(loc=10, scale=1).cdf(support)
    exponentials = norm(loc=10, scale=1).cdf(support)

    return fn(gaussians, exponentials), gaussians, exponentials


def plot_histograms(ps, xs, ys, test):
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


def plot(ps, xs, ys, support, test):
    mean_p = np.mean(ps)
    print("mean p = {}".format(mean_p))
    mean_x = np.mean(xs, axis=0)
    mean_y = np.mean(ys, axis=0)
    print(mean_x)
    print(mean_y)
    plt.subplot(211)
    plt.title("Two distributions with probability {:.2f} of being the same using {}".format(mean_p, test))
    plt.plot(support, mean_x)
    plt.subplot(212)
    plt.plot(support, mean_y)
    plt.show()


def make_comparison(n_trials, fn):
    trials = list(map(lambda _: fn(), range(0, n_trials)))
    m = np.asarray([*trials], object)
    metrics, xs, ys = np.transpose(m)
    return metrics, xs, ys


def run(fn, support):
    def run_func():
        return fn(support)
    return run_func


def t_test(support, n_trials=100):
    metrics, xs, ys = make_comparison(n_trials, run(using_t_test, support))
    tps = np.array([*metrics])
    return tps[:, 1], xs, ys


def kolmogorov_smirnov(support, n_trials=100):
    ps, xs, ys = make_comparison(n_trials, run(using_kolmogorov_smirnov, support))
    return ps, xs, ys


if __name__ == "__main__":
    over = np.arange(1, 20, 0.2)
    ps, xs, ys = kolmogorov_smirnov(over, 100)
    plot(ps, xs, ys, over, "KS")
