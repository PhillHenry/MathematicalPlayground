import numpy as np
import random as random
import matplotlib.pyplot as plt


def row_mean_of(xs):
    totals = np.sum(xs, axis=1)
    return totals / np.shape(xs)[1]


def covariance_of(xs):
    r"""
    :param xs: array-like object of values
    :return: the covariance matrix
    """
    averaged = xs - row_mean_of(xs)
    m = np.dot(averaged, averaged.T)
    n = np.shape(xs)[1]
    return m / (n - 1)


if __name__ == "__main__":
    num_measurements = 10
    num_students = 100
    rows = []
    for i in range(num_students):
        measurements = np.random.normal(random.gauss(150, 10), 5, num_measurements)
        rows.append(measurements)
    m = np.asmatrix(np.stack(rows))
    print("m is of dimensions: ", np.shape(m))
    c = covariance_of(m)
    heatmap = plt.imshow(c, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title("Covariance matrix")
    plt.show()


