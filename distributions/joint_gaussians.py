import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def gaussian(xs, mu, s):
    return stats.norm.pdf(xs, loc=mu, scale=s)


def joint(mu1, mu2, o1, o2, xs, ys):
    return gaussian(xs, mu1, o1) * gaussian(ys, mu2, o2)


# graphics taken from
# https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
if __name__ == "__main__":
    support = np.arange(-3, 3, .2)
    xs, ys = np.meshgrid(support, support)
    zs = joint(0, 0, 1, 1, xs, ys)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(np.min(zs), np.max(zs))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
