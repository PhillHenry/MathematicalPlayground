import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def gaussian(xs, mu, sd):
    return stats.norm.pdf(xs, loc=mu, scale=sd)


def joint(mu1, mu2, sd1, sd2, xs, ys):
    r"""
    (mu1 = 0; mu2 = 0; sd1 = 1; sd2 = 2) creates a peak curve at the origin that is symmetric under rotation.
    Changing the mu values translate the peak
    Changing the standard deviations changes the slope on the X- or Y- axis and eliminates rotational symmetry.
    :param mu1: the mean of the first Gaussian
    :param mu2: the mean of the second Gaussian
    :param sd1: the standard deviation of the first Gaussian
    :param sd2: the standard deviation of the second Gaussian
    :param xs: array-like input to the first Gaussian
    :param ys: array-like input to the second Gaussian
    :return: a matrix like object that represents the values of the first Gaussian multiplied by the second
    """
    return gaussian(xs, mu1, sd1) * gaussian(ys, mu2, sd2)


# graphics taken from
# https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
if __name__ == "__main__":
    support = np.arange(-3, 3, .2)
    xs, ys = np.meshgrid(support, support)
    zs = joint(0, 0, 1, 1, xs, ys)  # note that changing the means moves the centre and the standard deviations the spread

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(np.min(zs), np.max(zs))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
