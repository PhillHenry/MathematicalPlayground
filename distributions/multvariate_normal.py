import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def gaussian(xs, mu, s):
    return stats.norm.pdf(xs, loc=mu, scale=s)


def bivariate_distn(mu1, mu2, s1, s2, range1, range2):
    x1 = np.linspace(min(range1), max(range1))
    x2 = np.linspace(min(range2), max(range2))
    p1 = gaussian(x1, mu1, s1)
    p2 = gaussian(x2, mu2, s2)
    return p1 * p2


if __name__ == "__main__":
    step = .2
    X = np.arange(-5., 5., step)
    Y = np.arange(-5., 5., step)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(gaussian(X, 0, 1) + gaussian(Y, 0, 1))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
