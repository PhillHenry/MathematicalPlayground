# see https://en.wikipedia.org/wiki/Covariance#Example
import numpy as np

if __name__ == "__main__":
    data = np.asmatrix([[0, 0.4, 0.1], [0.3, 0, 0.2]])
    c = np.cov(data)
    p = np.linalg.inv(c)
    print("Covariance\n", c)
    print("Precision\n", p)

