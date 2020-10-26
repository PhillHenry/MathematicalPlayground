import matplotlib.pyplot as plt
import numpy as np

import graphics.files as f
import matrix.matrices as mat

# see https://stephens999.github.io/fiveMinuteStats/normal_markov_chain.html


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
    f.save_plot("/tmp/random_work.png")

    # heat map of the covariance of a
    c = np.cov(a)
    heatmap = plt.imshow(c, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title("Covariance matrix")
    f.save_plot("/tmp/covariance.png")

    # heat map of precision matrix
    p = np.linalg.inv(c)  # condition the covariance matrix
    # p = pd_inv(c)
    heatmap = plt.imshow(p[:10, :10], cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title("Precision matrix")
    f.save_plot("/tmp/precision.png")


