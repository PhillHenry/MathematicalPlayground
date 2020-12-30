import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt


def bootstramp(data, n, n_samples):
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
    n_boots = 1000
    data = np.random.normal(mu, sd, n)
    boot = bootstramp(data, n_boots, n)

    print("sample mean: %20f" % np.mean(data))
    print("sample sd:   %20f" % np.std(data))

    boot_means = [np.mean(x) for x in boot]
    boot_sd = [np.std(x) for x in boot]
    print("bootstrap mean mean: %12f" % np.mean(boot_means))
    print("bootstrap mean sd:   %12f" % np.mean(boot_sd))

    plt.hist(boot_means, bins=int(n_boots/30))
    # print("boot means: ", boot_means)
    plt.show()

    #print('Bootstrap Sample: %s' % boot)
    # out of bag observations
    #oob = [x for x in data if x not in boot]