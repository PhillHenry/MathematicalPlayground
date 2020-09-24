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

