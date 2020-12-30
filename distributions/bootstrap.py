import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import math


def bootstrap(data, n, n_samples):
    xs = []
    for _ in range(n):
        # see https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
        boot = resample(data, replace=True, n_samples=n_samples)
        xs.append(boot)
    return xs


if __name__ == "__main__":
    mu = 0.
    sd = 1.
    n = 25
    n_boots = 10000
    data = np.random.normal(mu, sd, n)
    boot = bootstrap(data, n_boots, n)

    print("sample mean: %20f" % np.mean(data))
    print("sample sd:   %20f" % np.std(data))

    boot_means = [np.mean(x) for x in boot]
    boot_sd = [np.std(x) for x in boot]
    print("bootstrap mean mean: %12f" % np.mean(boot_means))
    print("bootstrap mean sd:   %12f" % np.mean(boot_sd))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.hist(data, bins='auto')
    ax1.set_title('Actual data for %d points' % n)
    ax2.hist(boot_means, bins=int(math.log10(n_boots) * 10))
    ax2.set_title('Bootstrapping with %d iterations' % n_boots)

    # print("boot means: ", boot_means)
    plt.savefig("/tmp/boostrapping.png")
    plt.show()
