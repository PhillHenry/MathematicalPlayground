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
    return Comparison(summarize(granular), summarize(chunked))


def mean_and_std_of(xs: list) -> str:
    return "mean RMSE = {:>10f}, std = {:>10f}".format(np.mean(xs), np.std(xs))


if __name__ == "__main__":
    n_observations = 1000
    rmse_chunked = []
    rmse_granular = []
    for _ in range(n_observations):
        comparison = compare()
        rmse_chunked.append(comparison.chunked.rmse)
        rmse_granular.append(comparison.granular.rmse)
    print(f"In {n_observations} observations:")
    print(f"Chunks:   {mean_and_std_of(rmse_chunked)}")
    print(f"Granular: {mean_and_std_of(rmse_granular)}")
    # the std dev of the RMSE is larger for chunks because the underlying numbers are larger
    # however, the error is (roughly) the same
    plt.plot(range(n_observations), rmse_chunked, color='r')
    plt.plot(range(n_observations), rmse_granular, color='b')
    plt.show()
