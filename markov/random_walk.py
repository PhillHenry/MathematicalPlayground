import numpy as np
import markov.matrices as mat
import matplotlib.pyplot as plt
import distributions.covariance as cov

# see https://stephens999.github.io/fiveMinuteStats/normal_markov_chain.html


def save_plot(filename):
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    n = 100
    means = np.array(n)
    means.fill(0.)
    z = np.random.normal(means, 1., n)
    a = mat.lower_left(n)
    x = np.dot(a, z)

    # line plot of X against i
    plt.plot(x)
    plt.ylabel('X')
    plt.xlabel("i")
    plt.title("Markov chain of Gaussians forming a random walk")
    save_plot("/tmp/random_work.png")

    # heat map of the covariance of a
    c = cov.covariance_of(np.asmatrix(x))
    heatmap = plt.imshow(c, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.show()


