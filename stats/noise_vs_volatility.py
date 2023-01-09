import numpy as np
import matplotlib.pyplot as plt
import math as math


class Summary:
    def __init__(self, std, mean, rmse):
        self.std = std
        self.mean = mean
        self.rmse = rmse


class Comparison:
    def __init__(self, granular: Summary, chunked: Summary):
        self.chunked = chunked
        self.granular = granular


def summarize(xs: list):
    mean = np.mean(xs)
    std = np.std(xs)
    rmse = math.sqrt(sum((xs - mean) ** 2))
    return Summary(mean, std, rmse)


def compare() -> Comparison:
    n_chunks = 14
    granular = np.random.normal(10, 2, n_chunks * 24)
    chunked = []
    chunks = np.split(granular, n_chunks)
    for chunk in chunks:
        chunked.append(np.sum(chunk))
    granular = granular[0:len(chunks)]
    return Comparison(summarize(granular), summarize(chunked))


def mean_and_std_of(xs: list) -> str:
    return "mean = {:>10f}, std = {:>10f}".format(np.mean(xs), np.std(xs))


def make_comparisons(n_observations: int):
    comparisons = []
    for _ in range(n_observations):
        comparisons.append(compare())
    return comparisons


def do_compare(n_observations: int, summary_to_metric_fn):
    comparisons = make_comparisons(n_observations)
    chunked = []
    granular = []
    for comparison in comparisons:
        chunked.append(summary_to_metric_fn(comparison.chunked))
        granular.append(summary_to_metric_fn(comparison.granular))
    return chunked, granular


def std_dev_to_mean_ratio_of(x: Summary) -> float:
    return x.rmse / x.mean


def do_experiment():
    n_observations = 1000
    chunked, granular = do_compare(n_observations, std_dev_to_mean_ratio_of)
    print(f"In {n_observations} observations:")
    print(f"Chunks:   {mean_and_std_of(chunked)}")
    print(f"Granular: {mean_and_std_of(granular)}")
    # the std dev of the RMSE is larger for chunks because the underlying numbers are larger
    # however, the error is (roughly) the same
    plt.plot(range(n_observations), chunked, color='r', label="chunked")
    plt.plot(range(n_observations), granular, color='b', label="granular")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    do_experiment()
