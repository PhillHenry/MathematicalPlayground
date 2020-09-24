import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def gaussian(xs, mu, s):
    return stats.norm.pdf(xs, loc=mu, scale=s)


if __name__ == "__main__":  # graphics taken from https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
    step = .2
    X = np.arange(-3, 3, step)
    Y = np.arange(-3, 3, step)
    X, Y = np.meshgrid(X, Y)
    Z = gaussian(X, 0, 1) * gaussian(Y, 0, 1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(0, 0.3)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
